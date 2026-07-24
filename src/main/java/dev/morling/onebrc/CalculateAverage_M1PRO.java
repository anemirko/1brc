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
import java.lang.foreign.Arena;
import java.lang.reflect.Field;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import sun.misc.Unsafe;

/**
 * 1BRC solution tuned for Apple Silicon (M1 Pro) instead of the reference
 * EPYC 7502P (Zen2) evaluation box used by the official leaderboard.
 *
 * v15: replaces the precomputed ~102-163 fixed 128MB segments (each handed
 * to an ExecutorService as one Future) with fine-grained work stealing off
 * one shared AtomicLong cursor and a plain Thread[] - matching
 * CalculateAverage_thomaswue's mechanism. Tried because the v14 measurement
 * showed a smaller CPU-seconds gap to thomaswue (~1.37x) than wall-clock gap
 * (~1.54x), and the effective-parallelism figures (CPU-seconds/wall-time)
 * made the reason concrete: thomaswue ~8.11x, this file only ~7.23x on the
 * same 8 cores - suggesting load imbalance/straggler tail latency, not raw
 * per-row cost, which argued for finer work-distribution granularity.
 *
 * That hypothesis did NOT hold up: a chunk-size sweep from 512KB to 128MB
 * found wall time monotonically WORSE as chunks got smaller (512KB: 2.87s,
 * ~6.9x effective parallelism) and monotonically better as chunks got
 * bigger, plateauing around 8-128MB (all ~2.17-2.23s, ~7.3-7.45x) - the
 * opposite of what "steal more finely to fix load imbalance" predicts.
 * thomaswue's own 2MB SEGMENT_SIZE, copied as a starting point, measured as
 * one of the *worst* points on this machine (2.34s, ~7.2x) - Intel-tuned,
 * like the 3-way interleaving width before it, and not a fit here either.
 * 32MB was the best measured point and is the default; see CHUNK_SIZE.
 *
 * The actual, smaller win came from a different source than hypothesized:
 * even at matching 128MB granularity, this simpler AtomicLong+Thread
 * dispatch beat the old ExecutorService/Future mechanism outright (~1.6%
 * faster wall-clock, 7.23x -> 7.34x effective parallelism) - task-queue and
 * Future.get() bookkeeping overhead, not segment size, was worth a small
 * amount. Combined with the 32MB default: 2.22s/16.05 CPU-s/7.23x (v14) ->
 * 2.18s/16.07 CPU-s/7.37x - CPU-seconds unchanged, wall-clock and effective
 * parallelism both modestly better, gap to thomaswue narrows from ~1.54x to
 * ~1.49x wall-clock (CPU-seconds gap ~1.35x, about the same as before).
 *
 * Design choice worth flagging explicitly: each claimed chunk still gets its
 * own freshly-allocated OffHeapTable (freed via extractRawAndFree() at the
 * end of that chunk), the same as v14 - NOT a persistent table reused across
 * every chunk a thread claims over its lifetime, unlike thomaswue's
 * per-thread Result[] (allocated once, reused for that thread's whole run).
 * A persistent table was the more direct match to thomaswue's own structure
 * and was considered, but OffHeapTable's slots store each name's location as
 * a 4-byte int offset relative to the table's own segmentBase (see slot
 * layout below) - correct only because, until now, that base was always a
 * single chunk's own start. A table reused across chunks scattered
 * throughout a 13GB file would need those offsets to survive a change of
 * base between chunks, which the current 4-byte-int/single-base-per-table
 * layout can't represent (13GB overflows a 4-byte offset from a single
 * fixed base; correctly supporting it needs either widening the offset to
 * 8 bytes or storing a base pointer per slot, both touching the exact
 * update()/slot-layout code the OffHeapTable-vs-thomaswue disassembly
 * comparison already found lean and clean) - out of scope for a
 * granularity-only change. The chunk-size sweep above is itself indirect
 * evidence this matters: wall time degrading sharply at small chunk
 * sizes tracks allocating+zeroing a fresh ~1MB OffHeapTable far more often
 * (up to ~26,000 times at 512KB chunks for this 13GB file, vs ~102 times at
 * the old fixed 128MB granularity) - exactly the cost a persistent table
 * would avoid. Worth a follow-up if finer-grained stealing is revisited.
 *
 * v14: maps the whole file exactly once, up front, via
 * FileChannel.map(..., Arena.global()) - matching CalculateAverage_thomaswue -
 * instead of processSegment() calling channel.map() separately for every
 * segment (~102-163 times for a 13GB file at MAX_SEGMENT_SIZE). Tried after
 * disassembly comparisons of the hot-path methods (finishRow/processRow,
 * OffHeapTable.update() vs thomaswue's findResult+record, 4-way vs 3-way
 * interleaving) all came back clean - no fixable per-row instruction-count
 * waste left to find, so this tests the one remaining structural difference:
 * per-segment page-table setup cost paid ~150 times instead of once. Each
 * worker still gets its own [start,end) row-aligned range from
 * computeSegments() exactly as before; the only change is that range is now
 * an offset into one shared mapping instead of the arguments to its own
 * mapping call. This also lets RawEntry's lifetime drop the old
 * keepAlive-list/reachabilityFence machinery entirely - Arena.global() keeps
 * the whole file mapped for the JVM's lifetime, so there's nothing per-segment
 * left to keep reachable. See the benchmark note on processSegment() for
 * whether this measured as a real win.
 *
 * v8: replaces the variable-length SWAR name-scanning loop with
 * CalculateAverage_thomaswue's fixed 16-byte-window technique - see the
 * comment on processRow() for the full rationale and for why a v7 attempt to
 * interleave the *old* loop-based scan 3-way made things worse rather than
 * better (the loop itself, not the lack of interleaving, was the problem).
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
 * the standard dataset, not once per segment. (As of v14, RawEntry's base
 * address points into the one whole-file mapping, kept alive automatically
 * for the JVM's lifetime by Arena.global() - no explicit keepAlive/
 * reachabilityFence needed any more; see the v14 note above.)
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
 *    consistently faster. As of v15, work distribution is genuine
 *    fine-grained stealing off a shared cursor (see CHUNK_SIZE), not just
 *    small fixed segments handed out up front: whichever cores we actually
 *    get, a fast thread just claims more chunks and a slow one claims
 *    fewer - nobody blocks waiting on a fixed, possibly-mis-assigned chunk.
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
 *    HeapByteBuffer); Unsafe.getByte/getLong(address) is one raw load. The
 *    file's native base address is obtained once, up front, via
 *    MemorySegment.address() (see main()'s v14 note) instead of paying that
 *    cost per access. Trade-off: unlike ByteBuffer, out-of-bounds access
 *    here is a JVM-crashing SIGSEGV, not a catchable exception - fine for
 *    well-formed input, not forgiving of truncated/malformed files.
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
 * Usage: java --enable-preview CalculateAverage_M1PRO measurements.txt
 * (--enable-preview is required for java.lang.foreign.Arena on this
 * project's target JDK; drop it once the project moves to JDK 22+, where
 * that API is no longer a preview feature)
 */
public class CalculateAverage_M1PRO {

    private static final long BROADCAST_01 = 0x0101010101010101L;
    private static final long BROADCAST_80 = 0x8080808080808080L;
    private static final long SEMI_PATTERN = 0x3B3B3B3B3B3B3B3BL; // ';' repeated 8x
    private static final long NEWLINE_PATTERN = 0x0A0A0A0A0A0A0A0AL; // '\n' repeated 8x

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

    // v8: used only to determine nameLen via the fixed 16-byte-window fast path in
    // processRow (see there) - 0 when the delimiter was found within the first 8
    // bytes (word1's own letterCount1 already covers the whole name), all-ones when
    // it wasn't (so word2's letterCount2 also needs to count toward nameLen).
    // Index 8 means "not found in word1 at all". Exact technique and array from
    // this repo's CalculateAverage_thomaswue.
    private static final long[] FAST_MASK2 = { 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0x00L, 0xFFFFFFFFFFFFFFFFL };

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

        try (FileChannel channel = FileChannel.open(Path.of(fileName), StandardOpenOption.READ)) {

            long fileSize = channel.size();
            // v14: map the whole file exactly once, up front - matching
            // CalculateAverage_thomaswue - instead of processSegment() mapping
            // its own sub-range separately for every segment. Arena.global()
            // keeps this mapping alive for the JVM's lifetime, so RawEntry's
            // raw addresses (computed as fileBase+offset below) stay valid
            // with no explicit keepAlive/reachabilityFence needed - see class
            // Javadoc.
            long fileBase = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, Arena.global()).address();
            long fileEnd = fileBase + fileSize;
            int numThreads = performanceCoreCount();

            // v15: fine-grained work stealing off one shared cursor, matching
            // CalculateAverage_thomaswue, instead of precomputing ~102-163
            // fixed 128MB segments and handing each to the ExecutorService as
            // one Future - see class Javadoc for the rationale (measured
            // effective-parallelism gap vs thomaswue) and the chunk-size
            // sweep results.
            AtomicLong cursor = new AtomicLong(fileBase);
            Thread[] threads = new Thread[numThreads];
            List<RawEntry>[] threadResults = new List[numThreads];
            for (int t = 0; t < numThreads; t++) {
                final int index = t;
                threads[t] = new Thread(() -> {
                    List<RawEntry> collected = new ArrayList<>();
                    while (true) {
                        long current = cursor.addAndGet(CHUNK_SIZE) - CHUNK_SIZE;
                        if (current >= fileEnd) {
                            break;
                        }
                        long chunkEnd = findNextNewlineInFile(Math.min(fileEnd - 1, current + CHUNK_SIZE));
                        long chunkStart = (current == fileBase) ? current : findNextNewlineInFile(current) + 1;
                        if (chunkStart >= chunkEnd) {
                            // Degenerate claim: this thread's slice landed entirely
                            // inside a row its predecessor already owns (possible
                            // when CHUNK_SIZE is smaller than one row, or right at
                            // EOF) - nothing of this thread's own to process.
                            continue;
                        }
                        try {
                            collected.addAll(processSegment(fileBase, chunkStart - fileBase, chunkEnd - fileBase));
                        }
                        catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                    threadResults[index] = collected;
                });
                threads[t].start();
            }
            for (Thread thread : threads) {
                thread.join();
            }

            // Single global merge, raw-byte comparisons only. A String is
            // materialized at most once per genuinely unique station name
            // (not once per segment) - see class Javadoc.
            GlobalTable global = new GlobalTable();
            for (List<RawEntry> results : threadResults) {
                for (RawEntry e : results) {
                    global.merge(e);
                }
            }
            merged = global.toResults();
        }

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

    // v15: chunk size for the shared-cursor work-stealing loop in main() -
    // see class Javadoc for the sweep that picked this default. thomaswue's
    // own SEGMENT_SIZE (2MB, tuned for an Intel i9-13900K) was the starting
    // point, not assumed optimal here - and measured out to be a bad fit: a
    // sweep from 512KB to 128MB found wall time monotonically *improving* as
    // chunks got bigger (2MB: 2.34s -> 32MB: 2.17s), the opposite of what
    // "steal more finely to fix load imbalance" predicts. 32MB was the best
    // measured point (a shallow plateau from ~8MB-128MB, all within ~2% of
    // each other). Overridable via -DM1PRO.chunkSize=N for further tuning.
    private static final long CHUNK_SIZE = Long.getLong("M1PRO.chunkSize", 32L * 1024 * 1024);

    /**
     * Finds the next '\n' at or after the absolute address {@code pos},
     * unbounded - matching CalculateAverage_thomaswue's nextNewLine. Safe to
     * read slightly past the file's true end because the whole file is one
     * mmap (see main()'s v14 note): the OS zero-pads the final partial page,
     * and every caller here first clamps {@code pos} to at most fileEnd-1
     * (see main()), so the 8-byte read this does never crosses into an
     * unmapped page. Unlike nextNewLineBounded (used inside a claimed
     * chunk), this has no separate `limit` - the "whole file" bound already
     * comes from the caller-supplied Math.min(fileEnd-1, ...) clamp.
     */
    private static long findNextNewlineInFile(long pos) {
        while (true) {
            long word = UNSAFE.getLong(pos);
            long input = word ^ NEWLINE_PATTERN;
            long mask = (input - BROADCAST_01) & ~input & BROADCAST_80;
            if (mask != 0) {
                return pos + (Long.numberOfTrailingZeros(mask) >>> 3);
            }
            pos += 8;
        }
    }

    // Overridable via -DM1PRO.interleave=false for A/B testing against
    // the plain single-cursor sweep - see processSegment.
    private static final boolean INTERLEAVE = !"false".equals(System.getProperty("M1PRO.interleave"));

    // v12: thomaswue's 3-way interleaving width was tuned for an Intel i9-13900K
    // (per his file header comment), not for Apple Silicon. M1 Pro's core is wider
    // (more physical registers, wider decode) than contemporary x86 cores. A paired
    // A/B (8 rounds, both hand-unrolled with named locals to rule out array-indexing
    // overhead as a confound - see processSegmentInterleavedN's Javadoc for why that
    // matters) found width=4 winning 6/8 rounds, ~0.5% faster on average - small but
    // consistent, so it's the default here. Overridable via -DM1PRO.interleaveWidth=N;
    // 3 and 4 both route to their own hand-unrolled path (see processSegment /
    // processSegmentInterleaved4), other values go through the general array-based
    // path, which is *not* a clean comparison against either hand-unrolled path.
    private static final int INTERLEAVE_WIDTH = Integer.getInteger("M1PRO.interleaveWidth", 4);

    private static List<RawEntry> processSegment(long fileBase, long start, long end) throws Exception {
        long base = fileBase + start;
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
            return table.extractRawAndFree();
        }

        if (INTERLEAVE_WIDTH == 4) {
            return processSegmentInterleaved4(base, limit, table);
        }

        if (INTERLEAVE_WIDTH != 3) {
            return processSegmentInterleavedN(base, limit, table);
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

        return table.extractRawAndFree();
    }

    /**
     * v12: hand-unrolled 4-way interleaving, named locals (pos1..pos4), matching
     * the exact style of the 3-way block above - added specifically because the
     * general array-based N-way path below was found to carry its own overhead
     * (even at width=2, slower than the hand-unrolled 3-way baseline), confounding
     * any real signal about whether a wider width helps on M1 Pro's core. This
     * isolates the width variable from that array-indexing/JIT-scheduling overhead.
     */
    private static List<RawEntry> processSegmentInterleaved4(long base, int limit, OffHeapTable table) {
        int dist = limit / 4;
        int mid1 = nextNewLineBounded(base, dist, limit);
        int mid2 = nextNewLineBounded(base, dist * 2, limit);
        int mid3 = nextNewLineBounded(base, dist * 3, limit);

        int pos1 = 0;
        int end1 = mid1;
        int pos2 = mid1 + 1;
        int end2 = mid2;
        int pos3 = mid2 + 1;
        int end3 = mid3;
        int pos4 = mid3 + 1;
        int end4 = limit;

        while (pos1 < end1 && pos2 < end2 && pos3 < end3 && pos4 < end4) {
            pos1 = processRow(base, limit, pos1, table);
            pos2 = processRow(base, limit, pos2, table);
            pos3 = processRow(base, limit, pos3, table);
            pos4 = processRow(base, limit, pos4, table);
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
        while (pos4 < end4) {
            pos4 = processRow(base, limit, pos4, table);
        }

        return table.extractRawAndFree();
    }

    /**
     * v12: general N-way interleaving, for the -DM1PRO.interleaveWidth=N sweep -
     * see the field Javadoc. Splits the segment into N roughly equal, row-aligned
     * parts and round-robins processRow() across all N cursors per outer iteration,
     * same principle as the hand-unrolled 3-way version above, just array-indexed
     * to support a runtime-chosen width instead of 3 named locals.
     */
    private static List<RawEntry> processSegmentInterleavedN(long base, int limit, OffHeapTable table) {
        int n = INTERLEAVE_WIDTH;
        int[] pos = new int[n];
        int[] segEnd = new int[n];
        long dist = limit / (long) n;
        int prevEnd = -1;
        for (int k = 0; k < n; k++) {
            pos[k] = prevEnd + 1;
            segEnd[k] = (k == n - 1) ? limit : nextNewLineBounded(base, (int) (dist * (k + 1)), limit);
            prevEnd = segEnd[k];
        }

        while (true) {
            for (int k = 0; k < n; k++) {
                if (pos[k] >= segEnd[k]) {
                    // At least one cursor is out of rows - fall through to the
                    // per-cursor tail loops below for whichever ones still have work.
                    for (int t = 0; t < n; t++) {
                        while (pos[t] < segEnd[t]) {
                            pos[t] = processRow(base, limit, pos[t], table);
                        }
                    }
                    return table.extractRawAndFree();
                }
            }
            for (int k = 0; k < n; k++) {
                pos[k] = processRow(base, limit, pos[k], table);
            }
        }
    }

    private static long findDelimiter(long word) {
        long input = word ^ SEMI_PATTERN;
        return (input - BROADCAST_01) & ~input & BROADCAST_80;
    }

    /**
     * Parses one "name;value\n" row starting at {@code i} into {@code table}
     * and returns the position just past it. {@code limit} is always the
     * segment's own true end (not the calling cursor's sub-range end) - as of
     * v14 all segments share one whole-file mapping (see main()), so this
     * bound is no longer about avoiding an out-of-bounds read past a
     * segment-local mapping; it's still required for correctness of work
     * partitioning, so a cursor never scans into a neighboring segment's rows
     * (which another thread's OffHeapTable owns).
     *
     * v8: reads the first 16 bytes unconditionally and locates the ';' via a
     * single trailing-zero-count on a combined delimiter mask, instead of
     * looping 8 bytes at a time - credit: CalculateAverage_thomaswue. A v7
     * attempt to interleave the old loop-based scan 3-way across cursors
     * made things measurably worse (more CPU-seconds, not fewer): unrolling
     * a *loop* with an unpredictable trip count 3x creates three independent
     * loop-back branches for the scheduler to juggle, unlike thomaswue's
     * fixed, loop-free 16-byte window, which is what actually made his
     * interleaving cheap. This replaces the loop for the common case (name
     * <=16 bytes); the loop remains only as a fallback for the rare longer
     * names, or when too close to the segment's own end to safely read 16
     * bytes unconditionally. Unlike thomaswue's own fast path, the masked
     * word used for hashing/comparison is computed the same way (via
     * finishRow, below) regardless of which path determined nameLen - the
     * fast path only determines nameLen faster here, it doesn't reuse
     * thomaswue's exact (delimiter-inclusive) word masking, so there's no
     * risk of the same station producing inconsistent hash/comparison words
     * depending on which path a given occurrence happens to take.
     */
    private static int processRow(long base, int limit, int i, OffHeapTable table) {
        int nameStart = i;

        if (i + 16 <= limit) {
            long word = UNSAFE.getLong(base + i);
            long word2 = UNSAFE.getLong(base + i + 8);
            long delimiterMask = findDelimiter(word);
            long delimiterMask2 = findDelimiter(word2);
            if ((delimiterMask | delimiterMask2) != 0) {
                int letterCount1 = Long.numberOfTrailingZeros(delimiterMask) >>> 3;
                int letterCount2 = Long.numberOfTrailingZeros(delimiterMask2) >>> 3;
                long sel = FAST_MASK2[letterCount1];
                int nameLen = (int) (letterCount1 + (letterCount2 & sel));
                int afterDelimiter = nameStart + nameLen + 1;
                return finishRow(base, limit, nameStart, nameLen, afterDelimiter, word, word2, table);
            }
        }

        // Slow path: delimiter beyond the 16-byte window, or too close to
        // the segment's own end to safely read 16 bytes unconditionally.
        int j = nameStart;
        while (j + 8 <= limit) {
            long word = UNSAFE.getLong(base + j);
            long xored = word ^ SEMI_PATTERN;
            long hasZero = (xored - BROADCAST_01) & ~xored & BROADCAST_80;
            if (hasZero != 0) {
                break;
            }
            j += 8;
        }
        byte b;
        while ((b = UNSAFE.getByte(base + j)) != ';') {
            j++;
        }
        int nameLen = j - nameStart;
        long word = UNSAFE.getLong(base + nameStart);
        long word2 = UNSAFE.getLong(base + nameStart + 8);
        return finishRow(base, limit, nameStart, nameLen, j + 1, word, word2, table);
    }

    /**
     * Shared tail for both processRow() paths: masks word/word2 down to the
     * name's true length (same convention regardless of which path found
     * nameLen), derives a hash from the masked words, parses the number, and
     * updates the table. {@code i} is the position right after the ';'.
     */
    private static int finishRow(long base, int limit, int nameStart, int nameLen, int i, long word, long word2, OffHeapTable table) {
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
        int hash = finishName((int) combined ^ (int) (combined >>> 32));

        int value;
        if (i + 8 <= limit) {
            // Branchless number parse, adapted from CalculateAverage_thomaswue's
            // convertIntoNumber (itself credited there to Quan Anh Mai). JFR-guided
            // disassembly comparison (hsdis, C2-compiled hot path) found this was
            // the single biggest structural gap between the two implementations:
            // the byte-by-byte loop below compiles to a genuine data-dependent
            // conditional branch per digit character, while this version is a
            // fixed-latency shift/mask/multiply with zero data-dependent branches.
            // Same i+8<=limit guard convention as processRow's own i+16<=limit
            // check - only the read here, not the fallback below, needs it, since
            // i+8<=limit also guarantees (by the row format invariant) that a real
            // '\n' follows the fractional digit, not just segment-mapping safety.
            long numberWord = UNSAFE.getLong(base + i);
            int decimalSepPos = Long.numberOfTrailingZeros(~numberWord & 0x10101000L);
            value = (int) convertIntoNumber(decimalSepPos, numberWord);
            i += (decimalSepPos >>> 3) + 3;
        }
        else {
            // Slow path: too close to the segment's own end to safely read 8
            // bytes unconditionally - same hazard, same convention, as
            // processRow's i+16<=limit guard on the name side.
            boolean neg = false;
            byte b = UNSAFE.getByte(base + i);
            if (b == '-') {
                neg = true;
                i++;
                b = UNSAFE.getByte(base + i);
            }
            value = 0;
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
        }

        table.update(nameStart, nameLen, hash, firstWord, secondWord, value);
        return i;
    }

    // Branchless ASCII-to-int conversion for values in [-99.9, 99.9] with
    // exactly one fractional digit - see finishRow's fast path for how
    // numberWord/decimalSepPos are derived. Verbatim technique from
    // CalculateAverage_thomaswue::convertIntoNumber (credited there to Quan
    // Anh Mai); reused as-is since both implementations target the same
    // official 1BRC value range/format.
    private static long convertIntoNumber(int decimalSepPos, long numberWord) {
        int shift = 28 - decimalSepPos;
        long signed = (~numberWord << 59) >> 63;
        long designMask = ~(signed & 0xFF);
        long digits = ((numberWord & designMask) << shift) & 0x0F000F0F00L;
        long absValue = ((digits * 0x640a0001) >>> 32) & 0x3FF;
        return (absValue ^ signed) - signed;
    }

    /**
     * Finds the next '\n' at or after {@code pos}, never reading past
     * {@code limit}. Unlike CalculateAverage_thomaswue's unbounded
     * nextNewLine, this stays bounded even though (as of v14) both now share
     * one whole-file mapping - the bound here is about not scanning into a
     * neighboring segment's rows (a work-partitioning correctness concern),
     * not about mapping safety.
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

    /**
     * One occupied slot's data, extracted from a segment's off-heap table
     * with no String decoding - just the raw (address,len) the name lives
     * at plus its aggregates. base+offset points into the one whole-file
     * mapping (see main()'s v14 note), kept alive for the JVM's lifetime by
     * Arena.global() - no per-segment reachability tracking needed.
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

        // v8: firstWord/secondWord/hash are now computed once by the caller
        // (processRow/finishRow) and passed in directly, instead of being
        // recomputed here from nameOffset on every call - the caller already
        // has word/word2 loaded from its own 16-byte read, so recomputing
        // them again here would just be a second, redundant read of the same
        // bytes. (storedFirst, storedSecond) == (0, 0) remains a universal
        // empty-slot sentinel for every slot regardless of name length - no
        // real station name can produce (0,0) here since its first byte is
        // always non-zero printable text. Names longer than 16 bytes still
        // need regionEquals() to confirm full equality beyond that
        // first-16-byte prefix - credit: this repo's CalculateAverage_thomaswue,
        // which uses the identical (0,0)-as-empty / packed-word-first structure.
        void update(int nameOffset, int nameLen, int hash, long firstWord, long secondWord, int value) {
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
                        && (!needsFullCompare || regionEquals(segmentBase + UNSAFE.getInt(addr + 16), segmentBase + nameOffset, nameLen))) {
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
         * returned entries, and stays valid for the JVM's lifetime via
         * Arena.global() (see main()'s v14 note).
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
