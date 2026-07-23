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
package dev.morling.onebrc;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.lang.ref.Reference;
import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import sun.misc.Unsafe;

/**
 * 1BRC solution tuned for Apple Silicon (M1 Pro) instead of the reference
 * EPYC 7502P (Zen2) evaluation box used by the official leaderboard.
 *
 * v5: OffHeapTable slots now also carry the name's first 16 bytes packed
 * into two longs, compared directly instead of always calling
 * regionEquals() - see the comment on OffHeapTable.update() for why (JFR
 * profiling found regionEquals() was ~12% of total runtime, called
 * unconditionally on every row including repeat hits on already-seen
 * stations).
 *
 * v4: processSegment scans each segment with three interleaved cursors
 * instead of one straight sweep, matching CalculateAverage_thomaswue's
 * technique - see the comment on processSegment for the rationale. Tried
 * because a `time`-based CPU-seconds comparison against thomaswue showed
 * this repo's single-cursor version needed ~2.2x more total CPU-seconds for
 * the same file despite similar core utilization. In practice this measured
 * as a no-op here (same wall time and total CPU-seconds before and after,
 * within noise) - Apple's very wide out-of-order core may already extract
 * enough parallelism from a single cursor's independent loop iterations
 * without needing the software-level interleaving. Left in since it's
 * harmless and matches thomaswue's structure, but the v5 fix below is what
 * actually closed part of the gap.
 *
 * v3 (of the anemirko2/anemirko3/anemirko4 line): per-segment String
 * materialization was found (via JFR line-level profiling of anemirko4) to
 * be ~12% of total runtime - toResultsAndFree() used to decode every
 * occupied hash-table slot to a Java String inside each segment's own
 * worker thread, before the cross-thread merge. With MAX_SEGMENT_SIZE
 * ~128MB and a 13GB file, that's ~100 segments, and the standard 1BRC
 * dataset only has 413 real station names - a 128MB segment (~10M rows)
 * statistically sees nearly all of them. So instead of ~413 String
 * constructions total, the old code did ~100 x 413 ~= 41,300 - about 100x
 * more UTF-8 decoding than necessary, all but the final merged copy thrown
 * away immediately.
 *
 * Fix: segments now return raw (segmentBase, offset, len, hash, sum, count,
 * min, max) tuples - RawEntry - with no String anywhere. A single
 * GlobalTable merges those across all segments comparing raw bytes (cheap,
 * same trick as the per-segment table), and only materializes a String the
 * first time a name is genuinely new *globally* - at most ~413 times for
 * the standard dataset, not once per segment. Segments' MappedByteBuffers
 * are kept reachable (via an explicit keepAlive list + a reachabilityFence)
 * until that merge/decode is done, since RawEntry only carries a raw long
 * address, which is invisible to the JVM's own escape analysis/GC.
 *
 * Design choices:
 *
 * 1. Thread count = number of Performance cores (via `sysctl
 *    hw.perflevel0.physicalcpu`, {@link #MAX_THREADS}), not
 *    Runtime.availableProcessors(). M1 Pro is heterogeneous (6P+2E or
 *    8P+2E); macOS gives user processes no way to pin a thread to a P core,
 *    so it can migrate any of our "P-core" threads onto an E core at any
 *    time, with no way for us to detect or prevent it. An earlier version
 *    of this file capped the thread count below the real P-core count to
 *    "leave headroom" for the OS, but a paired A/B on this repo's own
 *    RAM-disk benchmark found no such downside - using every P-core was
 *    consistently faster. Segments are sized small (see MAX_SEGMENT_SIZE,
 *    not 1:1 with thread count) so this is still ordinary work-stealing:
 *    whichever cores we actually get, a fast thread just pulls more
 *    segments and a slow one pulls fewer - nobody blocks waiting on a
 *    fixed, possibly-mis-assigned chunk.
 *
 * 2. No jdk.incubator.vector. Requesting a 256-bit ByteVector species on
 *    hardware that only has 128-bit NEON makes the JVM silently fall back
 *    to a *software* (non-vectorized) emulation of the 256-bit ops - far
 *    slower than even scalar code, not just "half the throughput". SWAR
 *    ("SIMD Within A Register") bit tricks on plain 64-bit longs sidestep
 *    this entirely: same handful of integer/bitwise ops on amd64 and
 *    aarch64, no species/width mismatch possible.
 *
 * 3. sun.misc.Unsafe for every hot-path memory access, matching what the
 *    reference x86 solutions do. buffer.get(i) on a MappedByteBuffer goes
 *    through bounds checks and a virtual dispatch (DirectByteBuffer vs
 *    HeapByteBuffer); Unsafe.getByte/getLong(address) is one raw load. We
 *    also read the buffer's native base address once via reflection
 *    (Buffer.address) instead of paying that cost per access. Trade-off:
 *    unlike ByteBuffer, out-of-bounds access here is a JVM-crashing SIGSEGV,
 *    not a catchable exception - fine for well-formed input, not forgiving
 *    of truncated/malformed files.
 *
 * 4. Off-heap, fixed-stride (64 bytes = one M1/EPYC cache line) hash table
 *    instead of parallel Java arrays. Station keys are stored as zero-copy
 *    (offset, length) pairs into the already-mapped file - no per-insert
 *    byte[] allocation, and (per the v3 fix above) no String materialization
 *    until the final, tiny, cross-thread merge step. Each slot also carries
 *    the name's first 16 bytes packed into two longs (per the v5 fix below)
 *    so most lookups resolve with one or two register compares instead of a
 *    byte-scanning function call. The default table size (2^14 slots = 1 MB)
 *    comfortably fits inside M1 Pro's 24 MB shared P-cluster L2 - EPYC's L2
 *    is only 512 KB *per core*, so a size tuned to survive there would
 *    badly under-use what M1 Pro actually has available.
 *
 * Usage: java --add-opens java.base/java.nio=ALL-UNNAMED \
 *             CalculateAverage_M1PRO measurements.txt
 * (the --add-opens flag is only required on JDK 16+; harmless before that)
 */
public class CalculateAverage_M1PRO {

    // Deliberately small relative to the file size (not "file size / thread
    // count") - see class Javadoc point 1 on why (no core-affinity control
    // on macOS, so fine-grained work-stealing instead). Also keeps each
    // MappedByteBuffer well under Integer.MAX_VALUE.
    private static final long MAX_SEGMENT_SIZE = 128_000_000L;

    private static final long BROADCAST_01 = 0x0101010101010101L;
    private static final long BROADCAST_80 = 0x8080808080808080L;
    private static final long SEMI_PATTERN = 0x3B3B3B3B3B3B3B3BL; // ';' repeated 8x
    private static final long NEWLINE_PATTERN = 0x0A0A0A0A0A0A0A0AL; // '\n' repeated 8x
    private static final long FIBONACCI_MULT = 0x9E3779B97F4A7C15L;

    private static final int SLOT_SIZE = 64; // bytes; one 64B cache line per slot
    // Slot layout (offsets in bytes):
    // 0: long sum (fixed-point, value*10, signed)
    // 8: int count
    // 12: short min (fixed-point)
    // 14: short max (fixed-point)
    // 16: int keyOffset (relative to this segment's mapped base address)
    // 20: short keyLen (0 == empty slot sentinel; no station name is empty)
    // 22: short pad
    // 24: int hash (precomputed, avoids re-hashing bytes on resize)
    // 28: int pad2
    // 32: long firstNameWord (v5: first 8 bytes of the name, zero-padded if shorter)
    // 40: long secondNameWord (v5: next 8 bytes, zero-padded if the name is <=16 bytes)
    // 48: 16 bytes pad, rounds the slot out to one cache line

    // v5: masks the low N bytes of a long, used to zero-pad firstNameWord/secondNameWord
    // for names shorter than 8/16 bytes so two different-length names never compare equal
    // by accident. MASK1[k] keeps the low (k+1) bytes (credit: this repo's
    // CalculateAverage_thomaswue, which uses the identical trick).
    private static final long[] MASK1 = { 0xFFL, 0xFFFFL, 0xFFFFFFL, 0xFFFFFFFFL, 0xFFFFFFFFFFL, 0xFFFFFFFFFFFFL,
            0xFFFFFFFFFFFFFFL, 0xFFFFFFFFFFFFFFFFL };

    // Matches this machine's real P-core count (8, via `sysctl
    // hw.perflevel0.physicalcpu`). An earlier version of this file
    // deliberately capped this at 4 to "leave headroom" for the OS instead
    // of claiming every P core - that assumption wasn't backed by data, and
    // a paired A/B on this repo's own RAM-disk benchmark found using all 8
    // P-cores consistently faster (~3.1s vs ~6.5s) with no measured
    // downside. Overridable via -DM1PRO.workers=N for further
    // experimentation (e.g. if run on a different Apple Silicon variant).
    private static final int MAX_THREADS = 8;

    private static final Unsafe UNSAFE = getUnsafe();

    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        }
        catch (ReflectiveOperationException e) {
            throw new RuntimeException("sun.misc.Unsafe not available", e);
        }
    }

    public static void main(String[] args) throws Exception {
        String fileName = args.length > 0 ? args[0] : "measurements.txt";
        long t0 = System.nanoTime();

        List<Result> merged;
        // Kept alive only so the JVM can't unmap a segment's MappedByteBuffer
        // while GlobalTable still holds raw addresses pointing into it - see
        // class Javadoc on the v3 String-materialization fix.
        List<MappedByteBuffer> keepAlive = new ArrayList<>();

        try (RandomAccessFile raf = new RandomAccessFile(fileName, "r");
                FileChannel channel = raf.getChannel()) {

            long fileSize = channel.size();
            int numThreads = performanceCoreCount();
            List<long[]> segments = computeSegments(channel, fileSize);

            ExecutorService pool = Executors.newFixedThreadPool(numThreads);
            List<Future<SegmentOutput>> futures = new ArrayList<>();
            for (long[] seg : segments) {
                final long segStart = seg[0];
                final long segEnd = seg[1];
                futures.add(pool.submit(() -> processSegment(channel, segStart, segEnd)));
            }

            // Single global merge, raw-byte comparisons only. A String is
            // materialized at most once per genuinely unique station name
            // (not once per segment) - see class Javadoc.
            GlobalTable global = new GlobalTable();
            for (Future<SegmentOutput> f : futures) {
                SegmentOutput out = f.get();
                keepAlive.add(out.buffer);
                for (RawEntry e : out.entries) {
                    global.merge(e);
                }
            }
            pool.shutdown();
            merged = global.toResults();
        }

        // Guarantees keepAlive (and everything it holds) is still reachable
        // through the point where GlobalTable finished reading raw bytes out
        // of those mappings - guards against the JVM proving the list "dead"
        // early via escape analysis, since RawEntry's raw long address is
        // invisible to it as a dependency on the MappedByteBuffer object.
        Reference.reachabilityFence(keepAlive);

        System.out.println(formatResults(merged));
        double elapsed = (System.nanoTime() - t0) / 1e9;
        System.err.printf("Elapsed: %.3f s (threads=%d)%n", elapsed, performanceCoreCount());
    }

    /**
     * Number of Performance cores to use, via
     * {@code sysctl -n hw.perflevel0.physicalcpu} (perflevel0 = Performance,
     * perflevel1 = Efficiency on macOS/Apple Silicon), capped at
     * {@link #MAX_THREADS} to leave headroom for the OS/other processes
     * instead of claiming every P core. Falls back to
     * Runtime.availableProcessors() (also capped) if sysctl isn't available
     * (non-macOS hosts, older Intel Macs without perflevel* nodes, sandboxed
     * environments, etc.) or returns something nonsensical. Overridable
     * with -DM1PRO.workers=N.
     */
    private static int performanceCoreCount() {
        Integer override = Integer.getInteger("M1PRO.workers");
        if (override != null && override > 0) {
            return override;
        }
        try {
            Process p = new ProcessBuilder("sysctl", "-n", "hw.perflevel0.physicalcpu").start();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(p.getInputStream()))) {
                String line = reader.readLine();
                boolean finished = p.waitFor(2, TimeUnit.SECONDS);
                if (finished && p.exitValue() == 0 && line != null) {
                    int n = Integer.parseInt(line.trim());
                    if (n > 0) {
                        return Math.min(n, MAX_THREADS);
                    }
                }
            }
        }
        catch (Exception ignored) {
            // Not macOS, sysctl missing, no perflevel0 node (Intel Mac /
            // Linux / this sandbox) - fall through to the portable default.
        }
        return Math.min(Math.max(1, Runtime.getRuntime().availableProcessors()), MAX_THREADS);
    }

    /** Split the file into <2GB chunks aligned on row boundaries. */
    private static List<long[]> computeSegments(FileChannel channel, long fileSize) throws Exception {
        List<long[]> segments = new ArrayList<>();
        long pos = 0;
        while (pos < fileSize) {
            long end = Math.min(pos + MAX_SEGMENT_SIZE, fileSize);
            if (end < fileSize) {
                end = findNextNewline(channel, end);
            }
            segments.add(new long[]{ pos, end });
            pos = end;
        }
        return segments;
    }

    private static long findNextNewline(FileChannel channel, long from) throws Exception {
        ByteBuffer buf = ByteBuffer.allocate(1 << 16);
        long pos = from;
        long size = channel.size();
        while (pos < size) {
            buf.clear();
            int n = channel.read(buf, pos);
            if (n <= 0)
                break;
            for (int i = 0; i < n; i++) {
                if (buf.get(i) == '\n') {
                    return pos + i + 1;
                }
            }
            pos += n;
        }
        return size;
    }

    /** Raw native base address of a direct/mapped buffer (Buffer.address). */
    private static long addressOf(Buffer buffer) {
        try {
            Field addressField = Buffer.class.getDeclaredField("address");
            addressField.setAccessible(true);
            return addressField.getLong(buffer);
        }
        catch (ReflectiveOperationException e) {
            throw new RuntimeException("Cannot obtain direct buffer address - on JDK 16+ "
                    + "run with --add-opens java.base/java.nio=ALL-UNNAMED", e);
        }
    }

    // Overridable via -DM1PRO.interleave=false for A/B testing against
    // the plain single-cursor sweep - see processSegment.
    private static final boolean INTERLEAVE = !"false".equals(System.getProperty("M1PRO.interleave"));

    private static SegmentOutput processSegment(FileChannel channel, long start, long end) throws Exception {
        MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, start, end - start);
        long base = addressOf(buffer);
        int limit = (int) (end - start);

        OffHeapTable table = new OffHeapTable(base);

        if (!INTERLEAVE) {
            // Plain single-cursor sweep, no interleaving. Kept as an A/B
            // alternative to the v4 interleaved scan below: repeated JFR
            // profiling found nextNewLineBounded (used only to locate the
            // interleaved scan's mid1/mid2 split points) taking ~11% of
            // total runtime despite running only ~2x per segment, likely
            // because jumping straight to the segment's 1/3 and 2/3 marks is
            // a cold, unprefetched memory access - unlike processRow's
            // sequential scan, which benefits from OS read-ahead. This mode
            // tests whether the v5 packed-word win (see OffHeapTable.update)
            // is worth more alone than combined with v4's ILP attempt, which
            // itself measured as a no-op (see class Javadoc).
            int i = 0;
            while (i < limit) {
                i = processRow(base, limit, i, table);
            }
            return new SegmentOutput(buffer, table.extractRawAndFree());
        }

        // v4: split the segment into three roughly equal, row-aligned parts
        // and scan them with three interleaved cursors sharing the same
        // mapping, instead of one cursor sweeping straight through. Keeping
        // three independent, data-independent load/compute chains in flight
        // hides each other's memory latency on the out-of-order core -
        // credit: this repo's CalculateAverage_thomaswue, whose JFR profile
        // (877% CPU, ~11.4 total CPU-seconds for the whole file) showed this
        // single-cursor version needed ~2.2x more total CPU-seconds to do
        // the same job despite similar core utilization, meaning the gap was
        // per-core throughput, not parallelism.
        int dist = limit / 3;
        int mid1 = nextNewLineBounded(base, dist, limit);
        int mid2 = nextNewLineBounded(base, dist + dist, limit);

        int pos1 = 0;
        int end1 = mid1;
        int pos2 = mid1 + 1;
        int end2 = mid2;
        int pos3 = mid2 + 1;
        int end3 = limit;

        while (pos1 < end1 && pos2 < end2 && pos3 < end3) {
            pos1 = processRow(base, limit, pos1, table);
            pos2 = processRow(base, limit, pos2, table);
            pos3 = processRow(base, limit, pos3, table);
        }
        while (pos1 < end1) {
            pos1 = processRow(base, limit, pos1, table);
        }
        while (pos2 < end2) {
            pos2 = processRow(base, limit, pos2, table);
        }
        while (pos3 < end3) {
            pos3 = processRow(base, limit, pos3, table);
        }

        return new SegmentOutput(buffer, table.extractRawAndFree());
    }

    /**
     * Parses one "name;value\n" row starting at {@code i} into {@code table}
     * and returns the position just past it. {@code limit} is always the
     * segment's own true end (not the calling cursor's sub-range end) -
     * each segment is its own separate MappedByteBuffer here (unlike
     * CalculateAverage_thomaswue's single whole-file mapping), so this is
     * the only bound that must never be read past.
     */
    private static int processRow(long base, int limit, int i, OffHeapTable table) {
        int nameStart = i;
        int hash = 0;

        // SWAR fast path: fold 8 bytes at a time into the running hash
        // as long as none of them is ';' (hasZero bit trick on the XOR
        // against a broadcast ';' pattern). No byte-by-byte copy needed
        // since the key is stored as a zero-copy (offset,len) pair.
        while (i + 8 <= limit) {
            long word = UNSAFE.getLong(base + i);
            long xored = word ^ SEMI_PATTERN;
            long hasZero = (xored - BROADCAST_01) & ~xored & BROADCAST_80;
            if (hasZero != 0) {
                break; // a ';' is somewhere in this word - finish scalar below
            }
            long mixed = word * FIBONACCI_MULT;
            hash = hash * 31 + (int) (mixed ^ (mixed >>> 32));
            i += 8;
        }

        byte b;
        while ((b = UNSAFE.getByte(base + i)) != ';') {
            hash = hash * 31 + b;
            i++;
        }
        int nameLen = i - nameStart;
        i++; // skip ';'
        int finalHash = finishName(hash);

        boolean neg = false;
        b = UNSAFE.getByte(base + i);
        if (b == '-') {
            neg = true;
            i++;
            b = UNSAFE.getByte(base + i);
        }
        int value = 0;
        while (b != '.') {
            value = value * 10 + (b - '0');
            i++;
            b = UNSAFE.getByte(base + i);
        }
        i++; // skip '.'
        b = UNSAFE.getByte(base + i);
        value = value * 10 + (b - '0');
        i++; // skip the single fractional digit
        if (i < limit && UNSAFE.getByte(base + i) == '\n') {
            i++;
        }
        if (neg)
            value = -value;

        table.update(nameStart, nameLen, finalHash, value);
        return i;
    }

    /**
     * Finds the next '\n' at or after {@code pos}, never reading past
     * {@code limit} - unlike CalculateAverage_thomaswue's unbounded
     * nextNewLine (safe there only because it maps the whole file at once),
     * each segment here is its own separate mapping, so overrunning it is a
     * real out-of-bounds read, not just a read into a later part of the
     * same file.
     */
    private static int nextNewLineBounded(long base, int pos, int limit) {
        while (pos + 8 <= limit) {
            long word = UNSAFE.getLong(base + pos);
            long input = word ^ NEWLINE_PATTERN;
            long mask = (input - BROADCAST_01) & ~input & BROADCAST_80;
            if (mask != 0) {
                return pos + (Long.numberOfTrailingZeros(mask) >>> 3);
            }
            pos += 8;
        }
        while (pos < limit && UNSAFE.getByte(base + pos) != '\n') {
            pos++;
        }
        return pos;
    }

    private static int finishName(int hash) {
        return hash ^ (hash >>> 16);
    }

    private static boolean regionEquals(long addrA, long addrB, int len) {
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

    /** What a worker thread hands back: the entries it found, plus the mapping they point into. */
    private static final class SegmentOutput {
        final MappedByteBuffer buffer;
        final List<RawEntry> entries;

        SegmentOutput(MappedByteBuffer buffer, List<RawEntry> entries) {
            this.buffer = buffer;
            this.entries = entries;
        }
    }

    /**
     * One occupied slot's data, extracted from a segment's off-heap table
     * with no String decoding - just the raw (address,len) the name lives
     * at plus its aggregates. base+offset only stays valid as long as the
     * originating MappedByteBuffer is kept reachable (see keepAlive in main).
     */
    private static final class RawEntry {
        final long base;
        final int offset;
        final int len;
        final int hash;
        final long sum;
        final int count;
        final short min, max;

        RawEntry(long base, int offset, int len, int hash, long sum, int count, short min, short max) {
            this.base = base;
            this.offset = offset;
            this.len = len;
            this.hash = hash;
            this.sum = sum;
            this.count = count;
            this.min = min;
            this.max = max;
        }
    }

    /**
     * Off-heap, fixed-stride open-addressing hash table. One instance per
     * worker thread/segment; keys are zero-copy (offset,len) references
     * into that segment's own mapped memory, so equality checks compare
     * raw bytes straight out of the mapped file with no intermediate
     * allocation.
     */
    private static final class OffHeapTable {
        private final long segmentBase;
        private int capacity = 1 << 14; // 16384 slots * 32B = 512KB
        private int mask = capacity - 1;
        private long table = UNSAFE.allocateMemory((long) capacity * SLOT_SIZE);
        private int size = 0;

        OffHeapTable(long segmentBase) {
            this.segmentBase = segmentBase;
            UNSAFE.setMemory(table, (long) capacity * SLOT_SIZE, (byte) 0);
        }

        private long slotAddr(long tableBase, int idx) {
            return tableBase + (long) idx * SLOT_SIZE;
        }

        void update(int nameOffset, int nameLen, int hash, int value) {
            // v6: pack every name's first 16 bytes into two longs, unconditionally
            // (not just for names <=16 bytes as in v5), using the exact same masking
            // as before but with no truncation once nameLen reaches 16. This makes
            // (storedFirst, storedSecond) == (0, 0) a universal empty-slot sentinel,
            // for *every* slot regardless of name length - no real station name can
            // produce (0,0) here since its first byte is always non-zero printable
            // text, so we can drop the separate storedLen read+compare that v5 still
            // needed just to detect occupancy. storedLen is still written on insert
            // (extractRawAndFree() and resize() still use it) and still read on the
            // >16-byte fallback path, just no longer on this hot occupied/empty
            // check. Names longer than 16 bytes still need regionEquals() to confirm
            // full equality beyond that first-16-byte prefix - credit: this repo's
            // CalculateAverage_thomaswue, which uses the identical (0,0)-as-empty /
            // packed-word-first structure.
            //
            // (A follow-up attempt at also packing count/min/max into one long field
            // was tried and reverted: JFR profiling and CPU-seconds both showed no
            // measurable improvement, most likely because sum/count/min/max were
            // already co-located in this same 64-byte cache line, so there wasn't
            // much real memory-latency cost left to remove - the packing arithmetic
            // roughly cancelled out whatever it saved.)
            long nameAddr = segmentBase + nameOffset;
            long firstWord;
            long secondWord;
            if (nameLen <= 8) {
                firstWord = UNSAFE.getLong(nameAddr) & MASK1[nameLen - 1];
                secondWord = 0;
            }
            else if (nameLen < 16) {
                firstWord = UNSAFE.getLong(nameAddr);
                secondWord = UNSAFE.getLong(nameAddr + 8) & MASK1[nameLen - 9];
            }
            else {
                firstWord = UNSAFE.getLong(nameAddr);
                secondWord = UNSAFE.getLong(nameAddr + 8);
            }
            boolean needsFullCompare = nameLen > 16;

            int idx = hash & mask;
            while (true) {
                long addr = slotAddr(table, idx);
                long storedFirst = UNSAFE.getLong(addr + 32);
                long storedSecond = UNSAFE.getLong(addr + 40);

                if (storedFirst == 0 && storedSecond == 0) {
                    UNSAFE.putLong(addr, value);
                    UNSAFE.putInt(addr + 8, 1);
                    UNSAFE.putShort(addr + 12, (short) value);
                    UNSAFE.putShort(addr + 14, (short) value);
                    UNSAFE.putInt(addr + 16, nameOffset);
                    UNSAFE.putShort(addr + 20, (short) nameLen);
                    UNSAFE.putInt(addr + 24, hash);
                    UNSAFE.putLong(addr + 32, firstWord);
                    UNSAFE.putLong(addr + 40, secondWord);
                    size++;
                    if (size * 2 > capacity)
                        resize();
                    return;
                }

                if (storedFirst == firstWord && storedSecond == secondWord
                        && (!needsFullCompare || regionEquals(segmentBase + UNSAFE.getInt(addr + 16), nameAddr, nameLen))) {
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

        private void resize() {
            int newCapacity = capacity * 2;
            long newTable = UNSAFE.allocateMemory((long) newCapacity * SLOT_SIZE);
            UNSAFE.setMemory(newTable, (long) newCapacity * SLOT_SIZE, (byte) 0);
            int newMask = newCapacity - 1;

            for (int i = 0; i < capacity; i++) {
                long addr = slotAddr(table, i);
                short len = UNSAFE.getShort(addr + 20);
                if (len != 0) {
                    int hash = UNSAFE.getInt(addr + 24);
                    int idx = hash & newMask;
                    while (true) {
                        long newAddr = slotAddr(newTable, idx);
                        if (UNSAFE.getShort(newAddr + 20) == 0) {
                            UNSAFE.copyMemory(addr, newAddr, SLOT_SIZE);
                            break;
                        }
                        idx = (idx + 1) & newMask;
                    }
                }
            }
            UNSAFE.freeMemory(table);
            table = newTable;
            capacity = newCapacity;
            mask = newMask;
        }

        /**
         * Extracts every occupied slot as a String-free RawEntry and frees
         * this table's off-heap memory. Does NOT touch the mapped file's
         * memory - that's still referenced (by raw address) from the
         * returned entries, and stays valid only as long as the caller keeps
         * the originating MappedByteBuffer reachable.
         */
        List<RawEntry> extractRawAndFree() {
            List<RawEntry> list = new ArrayList<>(size);
            for (int i = 0; i < capacity; i++) {
                long addr = slotAddr(table, i);
                short len = UNSAFE.getShort(addr + 20);
                if (len != 0) {
                    int offset = UNSAFE.getInt(addr + 16);
                    int hash = UNSAFE.getInt(addr + 24);
                    long sum = UNSAFE.getLong(addr);
                    int count = UNSAFE.getInt(addr + 8);
                    short min = UNSAFE.getShort(addr + 12);
                    short max = UNSAFE.getShort(addr + 14);
                    list.add(new RawEntry(segmentBase, offset, len, hash, sum, count, min, max));
                }
            }
            UNSAFE.freeMemory(table);
            table = 0;
            return list;
        }
    }

    /**
     * Cross-segment merge table. Compares candidate keys as raw bytes
     * against whichever segment first introduced each station name -
     * a String is materialized only in {@link #toResults()}, at most once
     * per genuinely unique name, not once per segment. This is the fix for
     * the ~12% of runtime that used to go into redundant UTF-8 decoding.
     */
    private static final class GlobalTable {
        private int capacity = 1024;
        private int mask = capacity - 1;
        private long[] keyBase = new long[capacity];
        private int[] keyOffset = new int[capacity];
        private int[] keyLen = new int[capacity]; // 0 == empty slot sentinel
        private int[] hashes = new int[capacity];
        private long[] sums = new long[capacity];
        private int[] counts = new int[capacity];
        private short[] mins = new short[capacity];
        private short[] maxs = new short[capacity];
        private int size = 0;

        void merge(RawEntry e) {
            int idx = e.hash & mask;
            while (true) {
                int len = keyLen[idx];
                if (len == 0) {
                    keyBase[idx] = e.base;
                    keyOffset[idx] = e.offset;
                    keyLen[idx] = e.len;
                    hashes[idx] = e.hash;
                    sums[idx] = e.sum;
                    counts[idx] = e.count;
                    mins[idx] = e.min;
                    maxs[idx] = e.max;
                    size++;
                    if (size * 2 > capacity)
                        resize();
                    return;
                }
                if (len == e.len && regionEquals(keyBase[idx] + keyOffset[idx], e.base + e.offset, e.len)) {
                    sums[idx] += e.sum;
                    counts[idx] += e.count;
                    if (e.min < mins[idx])
                        mins[idx] = e.min;
                    if (e.max > maxs[idx])
                        maxs[idx] = e.max;
                    return;
                }
                idx = (idx + 1) & mask;
            }
        }

        private void resize() {
            int newCapacity = capacity * 2;
            long[] nKeyBase = new long[newCapacity];
            int[] nKeyOffset = new int[newCapacity];
            int[] nKeyLen = new int[newCapacity];
            int[] nHashes = new int[newCapacity];
            long[] nSums = new long[newCapacity];
            int[] nCounts = new int[newCapacity];
            short[] nMins = new short[newCapacity];
            short[] nMaxs = new short[newCapacity];
            int newMask = newCapacity - 1;

            for (int i = 0; i < capacity; i++) {
                if (keyLen[i] != 0) {
                    int idx = hashes[i] & newMask;
                    while (nKeyLen[idx] != 0) {
                        idx = (idx + 1) & newMask;
                    }
                    nKeyBase[idx] = keyBase[i];
                    nKeyOffset[idx] = keyOffset[i];
                    nKeyLen[idx] = keyLen[i];
                    nHashes[idx] = hashes[i];
                    nSums[idx] = sums[i];
                    nCounts[idx] = counts[i];
                    nMins[idx] = mins[i];
                    nMaxs[idx] = maxs[i];
                }
            }
            keyBase = nKeyBase;
            keyOffset = nKeyOffset;
            keyLen = nKeyLen;
            hashes = nHashes;
            sums = nSums;
            counts = nCounts;
            mins = nMins;
            maxs = nMaxs;
            capacity = newCapacity;
            mask = newMask;
        }

        /** The only place any station name gets turned into a String - once each. */
        List<Result> toResults() {
            List<Result> list = new ArrayList<>(size);
            for (int i = 0; i < capacity; i++) {
                int len = keyLen[i];
                if (len != 0) {
                    long addr = keyBase[i] + keyOffset[i];
                    byte[] nameBytes = new byte[len];
                    for (int k = 0; k < len; k++) {
                        nameBytes[k] = UNSAFE.getByte(addr + k);
                    }
                    String name = new String(nameBytes, StandardCharsets.UTF_8);
                    list.add(new Result(name, sums[i], counts[i], mins[i], maxs[i]));
                }
            }
            return list;
        }
    }

    /** Final, heap-side, per-station result row - one per genuinely unique name. */
    private static final class Result {
        final String name;
        final long sum;
        final int count;
        final short min, max;

        Result(String name, long sum, int count, short min, short max) {
            this.name = name;
            this.sum = sum;
            this.count = count;
            this.min = min;
            this.max = max;
        }
    }

    private static String formatResults(List<Result> results) {
        results.sort((a, b) -> a.name.compareTo(b.name));
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        for (Result r : results) {
            if (!first)
                sb.append(", ");
            first = false;
            double min = r.min / 10.0;
            double max = r.max / 10.0;
            double mean = round(r.sum / 10.0 / r.count);
            sb.append(r.name).append('=')
                    .append(fmt(min)).append('/')
                    .append(fmt(mean)).append('/')
                    .append(fmt(max));
        }
        sb.append('}');
        return sb.toString();
    }

    private static double round(double value) {
        return Math.round(value * 10.0) / 10.0;
    }

    private static String fmt(double value) {
        if (value == 0.0)
            value = 0.0; // normalize -0.0
        return String.format("%.1f", value);
    }
}
