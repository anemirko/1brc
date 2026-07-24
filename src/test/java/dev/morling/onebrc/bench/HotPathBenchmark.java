/*
 *  Copyright 2023 The original authors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package dev.morling.onebrc.bench;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import sun.misc.Unsafe;

/**
 * Isolated per-call microbenchmark for CalculateAverage_M1PRO's and
 * CalculateAverage_thomaswue's hot-path methods, built to settle a question
 * two earlier attempts this session couldn't: disassembly (a few rounds back)
 * found M1PRO's OffHeapTable.update() compiles to fewer static instructions
 * than thomaswue's findResult+record combined (64 vs 111), yet the full
 * 1B-row program needs ~1.46x more CPU-seconds - a mismatch static
 * instruction counts alone can't explain (they can't see cache misses,
 * branch mispredictions, or load-to-use latency). A follow-up attempt to
 * measure this directly with nanoTime() bracketing inside the real 10-thread
 * program hit a different wall: nanoTime() itself got ~45x more expensive
 * per call under sustained concurrent contention (measured: ~12.6ns
 * single-threaded vs ~573ns at 10-way concurrency), contaminating the very
 * thing it was trying to measure. JMH sidesteps both: single-threaded per
 * fork (no contention), proper warmup/iteration statistics (no one-off
 * timing), the standard tool for exactly "per-call cost of a hot method".
 *
 * The core methods under test (OffHeapTable, finishRow, processRow,
 * findDelimiter, finishName, regionEquals, MASK1/FAST_MASK2 for M1PRO;
 * Result, Scanner, findResult, record, scanNumber, hashToIndex,
 * convertIntoNumber, findDelimiter, MASK1/MASK2 for thomaswue) are
 * deliberately duplicated here verbatim from the real files rather than
 * called via reflection - both because their originals are private (not
 * reachable from another top-level class even in the same package) and
 * because reflective invocation would itself add per-call overhead that
 * would contaminate a nanosecond-scale measurement, the exact class of
 * problem this benchmark exists to avoid. This is a read-only snapshot for
 * measurement purposes; it is not kept in sync automatically if the real
 * implementations change.
 */
public class HotPathBenchmark {

    // ------------------------------------------------------------------
    // Shared synthetic data: real "name;value\n" rows drawn from the exact
    // same 413-station list and generation algorithm as
    // dev.morling.onebrc.CreateMeasurements (uniform station pick, then
    // round(gaussian(mean, 10), 1 decimal)) - not an invented distribution.
    // A ~200MB off-heap buffer (not the full 13GB file) gives realistic
    // memory-locality characteristics for the row-scanning benchmarks
    // without needing the real dataset.
    // ------------------------------------------------------------------

    static final Unsafe UNSAFE = getUnsafe();

    static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        }
        catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static final class Station {
        final String name;
        final double meanTemp;

        Station(String name, double meanTemp) {
            this.name = name;
            this.meanTemp = meanTemp;
        }
    }

    static List<Station> loadStations() {
        List<Station> stations = new ArrayList<>(413);
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                HotPathBenchmark.class.getResourceAsStream("/bench/stations.txt"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank())
                    continue;
                int idx = line.lastIndexOf(';');
                stations.add(new Station(line.substring(0, idx), Double.parseDouble(line.substring(idx + 1))));
            }
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
        if (stations.size() != 413) {
            throw new IllegalStateException("Expected 413 stations, loaded " + stations.size());
        }
        return stations;
    }

    /** Builds the shared off-heap buffer + row offsets once per JMH fork. */
    static final class SyntheticData {
        final long base;
        final int limit;
        final int[] rowStarts; // offset of each row's first byte, relative to base
        final List<Station> stations;

        SyntheticData() {
            this.stations = loadStations();
            long targetBytes = 200L * 1024 * 1024;
            long addr = UNSAFE.allocateMemory(targetBytes + 4096); // small slack for the last row's overrun
            this.base = addr;
            ThreadLocalRandom rnd = ThreadLocalRandom.current();
            List<Integer> starts = new ArrayList<>();
            long pos = 0;
            while (pos < targetBytes) {
                Station s = stations.get(rnd.nextInt(stations.size()));
                double temp = Math.round(rnd.nextGaussian() * 10.0 * 10.0 + s.meanTemp * 10.0) / 10.0;
                // clamp to the format's supported range, matching the real
                // official format's -99.9..99.9 assumption used throughout
                // CalculateAverage_M1PRO/thomaswue
                if (temp > 99.9)
                    temp = 99.9;
                if (temp < -99.9)
                    temp = -99.9;
                byte[] nameBytes = s.name.getBytes(StandardCharsets.UTF_8);
                String valueStr = String.format(java.util.Locale.ROOT, "%.1f", temp);
                starts.add((int) pos);
                for (byte b : nameBytes) {
                    UNSAFE.putByte(addr + pos, b);
                    pos++;
                }
                UNSAFE.putByte(addr + pos, (byte) ';');
                pos++;
                for (int i = 0; i < valueStr.length(); i++) {
                    UNSAFE.putByte(addr + pos, (byte) valueStr.charAt(i));
                    pos++;
                }
                UNSAFE.putByte(addr + pos, (byte) '\n');
                pos++;
            }
            this.limit = (int) pos;
            this.rowStarts = new int[starts.size()];
            for (int i = 0; i < starts.size(); i++) {
                rowStarts[i] = starts.get(i);
            }
        }
    }

    // A single shared buffer across all @State classes in this fork - each
    // @State below wraps a *view* over it (different precomputed tuples /
    // pre-populated tables), not a separate buffer, so all benchmarks read
    // from identical underlying data.
    static final SyntheticData DATA = new SyntheticData();

    // ==================================================================
    // M1PRO side - duplicated verbatim from CalculateAverage_M1PRO.java
    // (post-v17: absolute keyAddr slot layout, phase-level interleaving is
    // unrelated to these per-row methods and not duplicated here).
    // ==================================================================

    static final long BROADCAST_01 = 0x0101010101010101L;
    static final long BROADCAST_80 = 0x8080808080808080L;
    static final long SEMI_PATTERN = 0x3B3B3B3B3B3B3B3BL;
    static final int SLOT_SIZE = 64;
    static final long[] MASK1 = { 0xFFL, 0xFFFFL, 0xFFFFFFL, 0xFFFFFFFFL, 0xFFFFFFFFFFL, 0xFFFFFFFFFFFFL,
            0xFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL };
    static final long[] FAST_MASK2 = { 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0xFFFFFFFFFFFFFFFFL };

    static long findDelimiterM1(long word) {
        long input = word ^ SEMI_PATTERN;
        return (input - BROADCAST_01) & ~input & BROADCAST_80;
    }

    static int finishNameM1(int hash) {
        return hash ^ (hash >>> 16);
    }

    static boolean regionEqualsM1(long addrA, long addrB, int len) {
        int i = 0;
        while (i + 8 <= len) {
            if (UNSAFE.getLong(addrA + i) != UNSAFE.getLong(addrB + i))
                return false;
            i += 8;
        }
        while (i < len) {
            if (UNSAFE.getByte(addrA + i) != UNSAFE.getByte(addrB + i))
                return false;
            i++;
        }
        return true;
    }

    static long convertIntoNumberM1(int decimalSepPos, long numberWord) {
        int shift = 28 - decimalSepPos;
        long signed = (~numberWord << 59) >> 63;
        long designMask = ~(signed & 0xFF);
        long digits = ((numberWord & designMask) << shift) & 0x0F000F0F00L;
        long absValue = ((digits * 0x640a0001) >>> 32) & 0x3FF;
        return (absValue ^ signed) - signed;
    }

    /** Verbatim copy of CalculateAverage_M1PRO$OffHeapTable, minus resize()
     * (never triggered here - 413 real stations stay well under this
     * table's 8192-entry resize threshold, same as the real program). */
    static final class OffHeapTableM1 {
        int capacity = 1 << 14;
        int mask = capacity - 1;
        long table = UNSAFE.allocateMemory((long) capacity * SLOT_SIZE);

        OffHeapTableM1() {
            UNSAFE.setMemory(table, (long) capacity * SLOT_SIZE, (byte) 0);
        }

        long slotAddr(int idx) {
            return table + (long) idx * SLOT_SIZE;
        }

        void update(long nameAddr, int nameLen, int hash, long firstWord, long secondWord, int value) {
            boolean needsFullCompare = nameLen > 16;
            int idx = hash & mask;
            while (true) {
                long addr = slotAddr(idx);
                long storedFirst = UNSAFE.getLong(addr + 32);
                long storedSecond = UNSAFE.getLong(addr + 40);

                if (storedFirst == 0 && storedSecond == 0) {
                    UNSAFE.putLong(addr, value);
                    UNSAFE.putInt(addr + 8, 1);
                    UNSAFE.putShort(addr + 12, (short) value);
                    UNSAFE.putShort(addr + 14, (short) value);
                    UNSAFE.putLong(addr + 16, nameAddr);
                    UNSAFE.putShort(addr + 24, (short) nameLen);
                    UNSAFE.putInt(addr + 28, hash);
                    UNSAFE.putLong(addr + 32, firstWord);
                    UNSAFE.putLong(addr + 40, secondWord);
                    return;
                }

                if (storedFirst == firstWord && storedSecond == secondWord
                        && (!needsFullCompare || regionEqualsM1(UNSAFE.getLong(addr + 16), nameAddr, nameLen))) {
                    UNSAFE.putLong(addr, UNSAFE.getLong(addr) + value);
                    UNSAFE.putInt(addr + 8, UNSAFE.getInt(addr + 8) + 1);
                    short min = UNSAFE.getShort(addr + 12);
                    if (value < min)
                        UNSAFE.putShort(addr + 12, (short) value);
                    short max = UNSAFE.getShort(addr + 14);
                    if (value > max)
                        UNSAFE.putShort(addr + 14, (short) value);
                    return;
                }
                idx = (idx + 1) & mask;
            }
        }

        int occupiedSlots() {
            int n = 0;
            for (int i = 0; i < capacity; i++) {
                if (UNSAFE.getShort(slotAddr(i) + 24) != 0)
                    n++;
            }
            return n;
        }
    }

    /** Verbatim copy of CalculateAverage_M1PRO.finishRow's fast path only -
     * the slow (segment-tail) path is never exercised by this synthetic
     * buffer's interior rows, same as the real program's steady state. */
    static int finishRowM1(long base, int nameStart, int nameLen, int i, long word, long word2, OffHeapTableM1 table) {
        long firstWord;
        long secondWord;
        if (nameLen <= 8) {
            firstWord = word & MASK1[nameLen - 1];
            secondWord = 0;
        }
        else if (nameLen < 16) {
            firstWord = word;
            secondWord = word2 & MASK1[nameLen - 9];
        }
        else {
            firstWord = word;
            secondWord = word2;
        }
        long combined = firstWord ^ secondWord;
        int hash = finishNameM1((int) combined ^ (int) (combined >>> 32));

        long numberWord = UNSAFE.getLong(base + i);
        int decimalSepPos = Long.numberOfTrailingZeros(~numberWord & 0x10101000L);
        int value = (int) convertIntoNumberM1(decimalSepPos, numberWord);
        i += (decimalSepPos >>> 3) + 3;

        table.update(base + nameStart, nameLen, hash, firstWord, secondWord, value);
        return i;
    }

    /** Verbatim copy of CalculateAverage_M1PRO.processRow's fast path only. */
    static int processRowM1(long base, int limit, int i, OffHeapTableM1 table) {
        int nameStart = i;
        long word = UNSAFE.getLong(base + i);
        long word2 = UNSAFE.getLong(base + i + 8);
        long delimiterMask = findDelimiterM1(word);
        long delimiterMask2 = findDelimiterM1(word2);
        int letterCount1 = Long.numberOfTrailingZeros(delimiterMask) >>> 3;
        int letterCount2 = Long.numberOfTrailingZeros(delimiterMask2) >>> 3;
        long sel = FAST_MASK2[letterCount1];
        int nameLen = (int) (letterCount1 + (letterCount2 & sel));
        int afterDelimiter = nameStart + nameLen + 1;
        return finishRowM1(base, nameStart, nameLen, afterDelimiter, word, word2, table);
    }

    // ==================================================================
    // thomaswue side - duplicated verbatim from CalculateAverage_thomaswue.java
    // ==================================================================

    static long findDelimiterTW(long word) {
        long input = word ^ 0x3B3B3B3B3B3B3B3BL;
        return (input - 0x0101010101010101L) & ~input & 0x8080808080808080L;
    }

    static final long[] MASK1TW = new long[]{ 0xFFL, 0xFFFFL, 0xFFFFFFL, 0xFFFFFFFFL, 0xFFFFFFFFFFL, 0xFFFFFFFFFFFFL,
            0xFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL };
    static final long[] MASK2TW = new long[]{ 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0xFFFFFFFFFFFFFFFFL };

    static long convertIntoNumberTW(int decimalSepPos, long numberWord) {
        int shift = 28 - decimalSepPos;
        long signed = (~numberWord << 59) >> 63;
        long designMask = ~(signed & 0xFF);
        long digits = ((numberWord & designMask) << shift) & 0x0F000F0F00L;
        long absValue = ((digits * 0x640a0001) >>> 32) & 0x3FF;
        return (absValue ^ signed) - signed;
    }

    static int hashToIndexTW(long hash, ResultTW[] results) {
        long hashAsInt = hash ^ (hash >>> 33) ^ (hash >>> 15);
        return (int) (hashAsInt & (results.length - 1));
    }

    static final class ResultTW {
        long firstNameWord, secondNameWord;
        short min = 999, max = -999;
        int count;
        long sum;
        long nameAddress;
    }

    static final class ScannerTW {
        long pos;
        final long end;

        ScannerTW(long start, long end) {
            this.pos = start;
            this.end = end;
        }

        long pos() {
            return pos;
        }

        void add(long delta) {
            pos += delta;
        }

        long getLong() {
            return UNSAFE.getLong(pos);
        }

        long getLongAt(long p) {
            return UNSAFE.getLong(p);
        }
    }

    static ResultTW newEntryTW(ResultTW[] results, long nameAddress, int tableIndex, int nameLength, ScannerTW scanner) {
        ResultTW r = new ResultTW();
        results[tableIndex] = r;
        int totalLength = nameLength + 1;
        r.firstNameWord = scanner.getLongAt(nameAddress);
        r.secondNameWord = scanner.getLongAt(nameAddress + 8);
        if (totalLength <= 8) {
            r.firstNameWord = r.firstNameWord & MASK1TW[totalLength - 1];
            r.secondNameWord = 0;
        }
        else if (totalLength < 16) {
            r.secondNameWord = r.secondNameWord & MASK1TW[totalLength - 9];
        }
        r.nameAddress = nameAddress;
        return r;
    }

    /** Verbatim copy of CalculateAverage_thomaswue.findResult's fast path only. */
    static ResultTW findResultTW(long initialWord, long initialDelimiterMask, long wordB, long delimiterMaskB, ScannerTW scanner, ResultTW[] results) {
        ResultTW existingResult;
        long word = initialWord;
        long delimiterMask = initialDelimiterMask;
        long hash;
        long nameAddress = scanner.pos();
        long word2 = wordB;
        long delimiterMask2 = delimiterMaskB;

        int letterCount1 = Long.numberOfTrailingZeros(delimiterMask) >>> 3;
        int letterCount2 = Long.numberOfTrailingZeros(delimiterMask2) >>> 3;
        long mask = MASK2TW[letterCount1];
        word = word & MASK1TW[letterCount1];
        word2 = mask & word2 & MASK1TW[letterCount2];
        hash = word ^ word2;
        existingResult = results[hashToIndexTW(hash, results)];
        scanner.add(letterCount1 + (letterCount2 & mask));
        if (existingResult != null && existingResult.firstNameWord == word && existingResult.secondNameWord == word2) {
            return existingResult;
        }

        int nameLength = (int) (scanner.pos() - nameAddress);
        int tableIndex = hashToIndexTW(hash, results);
        outer: while (true) {
            existingResult = results[tableIndex];
            if (existingResult == null) {
                existingResult = newEntryTW(results, nameAddress, tableIndex, nameLength, scanner);
            }
            int i = 0;
            for (; i < nameLength + 1 - 8; i += 8) {
                if (scanner.getLongAt(existingResult.nameAddress + i) != scanner.getLongAt(nameAddress + i)) {
                    tableIndex = (tableIndex + 31) & (results.length - 1);
                    continue outer;
                }
            }
            int remainingShift = (64 - ((nameLength + 1 - i) << 3));
            if (((scanner.getLongAt(existingResult.nameAddress + i) ^ (scanner.getLongAt(nameAddress + i))) << remainingShift) == 0) {
                break;
            }
            else {
                tableIndex = (tableIndex + 31) & (results.length - 1);
            }
        }
        return existingResult;
    }

    static long scanNumberTW(ScannerTW scanPtr) {
        long numberWord = scanPtr.getLongAt(scanPtr.pos() + 1);
        int decimalSepPos = Long.numberOfTrailingZeros(~numberWord & 0x10101000L);
        long number = convertIntoNumberTW(decimalSepPos, numberWord);
        scanPtr.add((decimalSepPos >>> 3) + 4);
        return number;
    }

    static void recordTW(ResultTW existingResult, long number) {
        if (number < existingResult.min) {
            existingResult.min = (short) number;
        }
        if (number > existingResult.max) {
            existingResult.max = (short) number;
        }
        existingResult.sum += number;
        existingResult.count++;
    }

    static int occupiedTW(ResultTW[] results) {
        int n = 0;
        for (ResultTW r : results) {
            if (r != null)
                n++;
        }
        return n;
    }

    // ==================================================================
    // JMH states
    // ==================================================================

    /** Precomputed (nameAddr, nameLen, hash, firstWord, secondWord) tuples,
     * one per sampled row, spread across the whole 200MB buffer - built
     * once at @Setup time by a full processRowM1 pre-scan (not guessed),
     * so the timed update()-only benchmark measures purely the steady-state
     * repeat-hit path against a table already holding all 413 stations. */
    @State(Scope.Thread)
    public static class M1TableState {
        OffHeapTableM1 table;
        long[] nameAddr;
        int[] nameLen;
        int[] hash;
        long[] firstWord;
        long[] secondWord;
        int[] value;
        int cursor;

        @Setup
        public void setup() {
            table = new OffHeapTableM1();
            int sampleEvery = 20;
            int n = DATA.rowStarts.length / sampleEvery;
            nameAddr = new long[n];
            nameLen = new int[n];
            hash = new int[n];
            firstWord = new long[n];
            secondWord = new long[n];
            value = new int[n];
            ThreadLocalRandom rnd = ThreadLocalRandom.current();
            int j = 0;
            for (int r = 0; r < DATA.rowStarts.length; r++) {
                int nameStart = DATA.rowStarts[r];
                long word = UNSAFE.getLong(DATA.base + nameStart);
                long word2 = UNSAFE.getLong(DATA.base + nameStart + 8);
                long delimiterMask = findDelimiterM1(word);
                long delimiterMask2 = findDelimiterM1(word2);
                int letterCount1 = Long.numberOfTrailingZeros(delimiterMask) >>> 3;
                int letterCount2 = Long.numberOfTrailingZeros(delimiterMask2) >>> 3;
                long sel = FAST_MASK2[letterCount1];
                int nLen = (int) (letterCount1 + (letterCount2 & sel));
                long fw, sw;
                if (nLen <= 8) {
                    fw = word & MASK1[nLen - 1];
                    sw = 0;
                }
                else if (nLen < 16) {
                    fw = word;
                    sw = word2 & MASK1[nLen - 9];
                }
                else {
                    fw = word;
                    sw = word2;
                }
                long combined = fw ^ sw;
                int h = finishNameM1((int) combined ^ (int) (combined >>> 32));
                // Insert into the table for every row (steady-state population,
                // matching the real program's full scan), but only *record* a
                // sampled subset as benchmark input tuples.
                table.update(DATA.base + nameStart, nLen, h, fw, sw, r);
                if (r % sampleEvery == 0 && j < n) {
                    nameAddr[j] = DATA.base + nameStart;
                    nameLen[j] = nLen;
                    hash[j] = h;
                    firstWord[j] = fw;
                    secondWord[j] = sw;
                    value[j] = rnd.nextInt(-999, 1000);
                    j++;
                }
            }
            int occupied = table.occupiedSlots();
            if (occupied != 413) {
                throw new IllegalStateException("M1TableState: expected 413 occupied slots after pre-population, got " + occupied);
            }
            cursor = 0;
        }

        long nextNameAddr() {
            cursor = (cursor + 1 == nameAddr.length) ? 0 : cursor + 1;
            return nameAddr[cursor];
        }
    }

    @State(Scope.Thread)
    public static class TWTableState {
        ResultTW[] results;
        long[] nameAddress;
        long[] initialWord;
        long[] initialDelimiterMask;
        long[] wordB;
        long[] delimiterMaskB;
        int cursor;

        @Setup
        public void setup() {
            results = new ResultTW[1 << 17];
            int sampleEvery = 20;
            int n = DATA.rowStarts.length / sampleEvery;
            nameAddress = new long[n];
            initialWord = new long[n];
            initialDelimiterMask = new long[n];
            wordB = new long[n];
            delimiterMaskB = new long[n];
            int j = 0;
            for (int r = 0; r < DATA.rowStarts.length; r++) {
                int nameStart = DATA.rowStarts[r];
                long addr = DATA.base + nameStart;
                long word = UNSAFE.getLong(addr);
                long word2 = UNSAFE.getLong(addr + 8);
                long dm1 = findDelimiterTW(word);
                long dm2 = findDelimiterTW(word2);
                ScannerTW scanner = new ScannerTW(addr, addr + 200);
                findResultTW(word, dm1, word2, dm2, scanner, results);
                if (r % sampleEvery == 0 && j < n) {
                    nameAddress[j] = addr;
                    initialWord[j] = word;
                    initialDelimiterMask[j] = dm1;
                    wordB[j] = word2;
                    delimiterMaskB[j] = dm2;
                    j++;
                }
            }
            int occupied = occupiedTW(results);
            if (occupied != 413) {
                throw new IllegalStateException("TWTableState: expected 413 occupied slots after pre-population, got " + occupied);
            }
            cursor = 0;
        }

        int next() {
            cursor = (cursor + 1 == nameAddress.length) ? 0 : cursor + 1;
            return cursor;
        }
    }

    /** For the finishRow-end-to-end benchmark: pre-scanned, correct
     * (nameStart, nameLen, afterDelimiter, word, word2) tuples, spread
     * across the whole buffer, plus a table already at steady state. */
    @State(Scope.Thread)
    public static class M1FinishRowState {
        OffHeapTableM1 table;
        int[] nameStart;
        int[] nameLen;
        int[] afterDelimiter;
        long[] word;
        long[] word2;
        int cursor;

        @Setup
        public void setup() {
            table = new OffHeapTableM1();
            int sampleEvery = 20;
            int n = DATA.rowStarts.length / sampleEvery;
            nameStart = new int[n];
            nameLen = new int[n];
            afterDelimiter = new int[n];
            word = new long[n];
            word2 = new long[n];
            int j = 0;
            for (int r = 0; r < DATA.rowStarts.length; r++) {
                int ns = DATA.rowStarts[r];
                long w = UNSAFE.getLong(DATA.base + ns);
                long w2 = UNSAFE.getLong(DATA.base + ns + 8);
                long dm1 = findDelimiterM1(w);
                long dm2 = findDelimiterM1(w2);
                int lc1 = Long.numberOfTrailingZeros(dm1) >>> 3;
                int lc2 = Long.numberOfTrailingZeros(dm2) >>> 3;
                long sel = FAST_MASK2[lc1];
                int nLen = (int) (lc1 + (lc2 & sel));
                int after = ns + nLen + 1;
                // populate table for steady state
                finishRowM1(DATA.base, ns, nLen, after, w, w2, table);
                if (r % sampleEvery == 0 && j < n) {
                    nameStart[j] = ns;
                    nameLen[j] = nLen;
                    afterDelimiter[j] = after;
                    word[j] = w;
                    word2[j] = w2;
                    j++;
                }
            }
            int occupied = table.occupiedSlots();
            if (occupied != 413) {
                throw new IllegalStateException("M1FinishRowState: expected 413 occupied slots, got " + occupied);
            }
            cursor = 0;
        }

        int next() {
            cursor = (cursor + 1 == nameStart.length) ? 0 : cursor + 1;
            return cursor;
        }
    }

    /** For the thomaswue end-to-end benchmark: same idea, pre-scanned
     * nameAddress values plus a steady-state Result[] table. */
    @State(Scope.Thread)
    public static class TWFinishRowState {
        ResultTW[] results;
        long[] nameAddress;
        int cursor;

        @Setup
        public void setup() {
            results = new ResultTW[1 << 17];
            int sampleEvery = 20;
            int n = DATA.rowStarts.length / sampleEvery;
            nameAddress = new long[n];
            int j = 0;
            for (int r = 0; r < DATA.rowStarts.length; r++) {
                long addr = DATA.base + DATA.rowStarts[r];
                long word = UNSAFE.getLong(addr);
                long word2 = UNSAFE.getLong(addr + 8);
                long dm1 = findDelimiterTW(word);
                long dm2 = findDelimiterTW(word2);
                ScannerTW scanner = new ScannerTW(addr, addr + 200);
                ResultTW existing = findResultTW(word, dm1, word2, dm2, scanner, results);
                long number = scanNumberTW(scanner);
                recordTW(existing, number);
                if (r % sampleEvery == 0 && j < n) {
                    nameAddress[j] = addr;
                    j++;
                }
            }
            int occupied = occupiedTW(results);
            if (occupied != 413) {
                throw new IllegalStateException("TWFinishRowState: expected 413 occupied slots, got " + occupied);
            }
            cursor = 0;
        }

        int next() {
            cursor = (cursor + 1 == nameAddress.length) ? 0 : cursor + 1;
            return cursor;
        }
    }

    // ==================================================================
    // Benchmarks
    // ==================================================================

    /** Isolates OffHeapTable.update() alone - the method the earlier
     * disassembly round compared against thomaswue's findResult+record. */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 5, time = 1)
    @Measurement(iterations = 5, time = 1)
    @Fork(1)
    public void m1pro_updateOnly(M1TableState s, Blackhole bh) {
        int c = (s.cursor + 1 == s.nameAddr.length) ? 0 : s.cursor + 1;
        s.cursor = c;
        s.table.update(s.nameAddr[c], s.nameLen[c], s.hash[c], s.firstWord[c], s.secondWord[c], s.value[c]);
        bh.consume(s.table);
    }

    /** Isolates thomaswue's findResult()+record() together - the fair
     * comparison unit established in the earlier disassembly round. */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 5, time = 1)
    @Measurement(iterations = 5, time = 1)
    @Fork(1)
    public void thomaswue_findResultAndRecord(TWTableState s, Blackhole bh) {
        int c = s.next();
        ScannerTW scanner = new ScannerTW(s.nameAddress[c], s.nameAddress[c] + 200);
        ResultTW existing = findResultTW(s.initialWord[c], s.initialDelimiterMask[c], s.wordB[c], s.delimiterMaskB[c], scanner, s.results);
        recordTW(existing, 123); // fixed value: this benchmark isolates lookup+record cost, not number parsing
        bh.consume(existing);
    }

    /** M1PRO finishRow() end-to-end: name masking + hash + branchless
     * number parse + update() call - full per-row tail, not just update(). */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 5, time = 1)
    @Measurement(iterations = 5, time = 1)
    @Fork(1)
    public void m1pro_finishRowEndToEnd(M1FinishRowState s, Blackhole bh) {
        int c = s.next();
        int result = finishRowM1(DATA.base, s.nameStart[c], s.nameLen[c], s.afterDelimiter[c], s.word[c], s.word2[c], s.table);
        bh.consume(result);
    }

    /** thomaswue's equivalent end-to-end sequence: delimiter scan + findResult
     * + scanNumber + record - same scope as finishRow above. */
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Warmup(iterations = 5, time = 1)
    @Measurement(iterations = 5, time = 1)
    @Fork(1)
    public void thomaswue_endToEnd(TWFinishRowState s, Blackhole bh) {
        int c = s.next();
        long addr = s.nameAddress[c];
        long word = UNSAFE.getLong(addr);
        long word2 = UNSAFE.getLong(addr + 8);
        long dm1 = findDelimiterTW(word);
        long dm2 = findDelimiterTW(word2);
        ScannerTW scanner = new ScannerTW(addr, addr + 200);
        ResultTW existing = findResultTW(word, dm1, word2, dm2, scanner, s.results);
        long number = scanNumberTW(scanner);
        recordTW(existing, number);
        bh.consume(existing);
    }
}
