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

import java.lang.foreign.Arena;
import java.lang.reflect.Field;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import sun.misc.Unsafe;

/**
 * 1BRC solution tuned for Apple Silicon (M1 Pro) instead of the reference
 * EPYC 7502P (Zen2) evaluation box used by the official leaderboard.
 *
 * v17: adds processSegmentPhased4, a phase-level interleaving alternative to
 * processSegmentInterleaved4 (row-level) - promoted to the width=4 default
 * after the A/B below; row-level stays available via
 * -DM1PRO.interleaveMode=row for further comparison. See
 * processSegmentPhased4's own Javadoc for the full rationale (re-testing
 * phase-level interleaving now that v8's fixed 16-byte-window scan has
 * removed the loop-unrolling problem that sank the original v7 attempt at
 * this). Two independent, cheap checks preceded this: (1) OffHeapTable's
 * open-addressing probe stride (+1) was measured, not just reasoned about -
 * a temporary counter found 1.0557 average probes per update() call (4.33%
 * of calls need 2+ probes) on a full run, real but modest overhead, not
 * followed up with a stride change here since it wasn't the subject of this
 * round; (2) the same probe counter re-run under phased mode measured
 * 1.0557 again, confirming phasing and probe/collision behavior are
 * independent, as expected (phasing only changes when table.update() fires
 * relative to other cursors, not the hash/insertion-order pattern that
 * determines collisions).
 *
 * Result: a real, if modest, win - unlike v16, which moved CPU-seconds and
 * wall-clock in opposite-or-flat directions (a wash), phasing moved both
 * together. 5-round A/B, same file, back-to-back: 2.046s/17.52 CPU-s/8.56x
 * (row-level) vs 1.970s/16.90 CPU-s/8.58x (phased) - about 3.5-3.7% less of
 * both wall-clock and CPU-seconds, not just better scheduling. Narrows the
 * gap to thomaswue from ~1.40x to ~1.35x wall-clock (~1.48x to ~1.43x
 * CPU-seconds). Correctness verified against all 12 sample files plus three
 * full-file diffs against thomaswue's reference output, in phased mode
 * specifically (row-level mode was already covered by every prior round's
 * verification, and stays available as the -DM1PRO.interleaveMode=row
 * override now that phased is the default).
 *
 * v16: gives each worker thread one persistent OffHeapTable for its entire
 * run, reused across every chunk it claims off the v15 AtomicLong cursor,
 * instead of a fresh table per chunk. Tried as a targeted test of a specific
 * hypothesis, not a guess: after v15 fixed effective parallelism (M1PRO
 * matched/exceeded thomaswue's CPU-seconds/wall-time ratio), the remaining
 * gap was ~48% more total CPU-seconds for the same job, and a table-lifecycle
 * profiling pass (nanoTime bracketing around OffHeapTable construction,
 * extractRawAndFree(), and GlobalTable.merge(), summed across all ~400
 * per-chunk tables at the 32MB default) found that combined overhead was
 * 0.25-0.42% of total CPU-seconds - conclusively ruling out construction/
 * extraction/merge cost as the cause. This change targets a different,
 * related hypothesis the profiling pass couldn't see: cache-miss cost from
 * every chunk hitting a cold, freshly-zeroed table, invisible to a
 * phase-based measurement because it's baked into the already-large
 * row-processing bucket, not a separate line item - matching thomaswue's own
 * per-thread Result[] reuse instead of allocating fresh per unit of work.
 *
 * Widened OffHeapTable's per-slot keyOffset from a 4-byte int (relative to
 * one chunk's own base) to an 8-byte absolute keyAddr, since a persistent
 * table now holds entries from many different chunks scattered across the
 * whole 13GB file with no single base they're all relative to any more (see
 * slot layout below) - fit inside the existing padding, so slots stay 64
 * bytes/one cache line. RawEntry and GlobalTable simplified the same way
 * (a single absolute address instead of a base+offset pair) as a direct,
 * required consequence, not scope creep.
 *
 * Result: a wash. 5-round A/B, same file, back-to-back: 2.032s/17.72
 * CPU-s/8.72x effective parallelism (v15) vs 2.032s/17.59 CPU-s/8.66x (v16) -
 * identical wall-clock, a ~0.7% CPU-seconds difference well inside this
 * session's noise floor. A follow-up attempt to isolate OffHeapTable.update()
 * specifically (same nanoTime-bracketing technique that worked cleanly for
 * the table-lifecycle profiling above) hit a genuine measurement wall rather
 * than a clean answer either way: timing every call (~13M+ times) caused
 * System.nanoTime() itself to become ~45x more expensive per call under
 * sustained 10-thread concurrent contention on this machine (measured
 * directly: ~12.6ns single-threaded vs ~573ns under 10-way concurrency),
 * which alone made the instrumented run 10x+ slower - not a logic bug (small
 * inputs ran instantly and correctly; jstack confirmed all 10 threads
 * genuinely RUNNABLE inside update(), not deadlocked). Sampling 1-in-997
 * calls avoided the hang but still produced a result contaminated by the
 * same contention (an extrapolated update()-only total that exceeded the
 * entire program's measured CPU budget for that run - not just noisy, but
 * arithmetically impossible), so no isolated per-method number is reported
 * either way. Combined with the wash on the real, uninstrumented wall-clock/
 * CPU-seconds numbers above, this closes off both table-related hypotheses
 * (construction overhead, ruled out by the earlier profiling; cache warmth,
 * a wash here) without a confirmed mechanism either way for the remaining
 * gap - kept anyway since a persistent table is the more direct structural
 * match to thomaswue and isn't a regression.
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
 * 1. Thread count = Runtime.availableProcessors() (all cores, P+E),
 *    matching CalculateAverage_thomaswue exactly - see workerThreadCount()
 *    for the A/B that overturned an earlier, unverified P-core-only design.
 *    Work distribution is fine-grained stealing off a shared cursor (see
 *    CHUNK_SIZE, v15 note below), not fixed segments handed out up front:
 *    whichever cores we actually get, a fast thread just claims more chunks
 *    and a slow one claims fewer - nobody blocks waiting on a fixed,
 *    possibly-mis-assigned chunk, which is also why letting the 2 E-cores
 *    in doesn't cost anything: a slower thread just finishes later, without
 *    blocking siblings on a chunk it was never handed in the first place.
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
    // 16: long keyAddr (v16: absolute address of the name's first byte -
    // base+nameOffset, computed once by the caller - widened from a
    // 4-byte offset relative to a single chunk's own base, because a
    // persistent per-thread table (see class Javadoc) now holds entries
    // from many different chunks scattered across the whole 13GB file,
    // with no single base they're all relative to any more)
    // 24: short keyLen (0 == empty slot sentinel; no station name is empty)
    // 26: short pad
    // 28: int hash (precomputed, avoids re-hashing bytes on resize)
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
            int numThreads = workerThreadCount();

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
                    // v16: one persistent OffHeapTable per thread, reused
                    // across every chunk it claims, instead of a fresh table
                    // per chunk - see class Javadoc for why (a table-lifecycle
                    // profiling pass ruled out construction/extraction/merge
                    // overhead as the cause of the CPU-seconds gap to
                    // thomaswue; this tests cache-locality from a warm,
                    // reused table instead, matching thomaswue's own
                    // per-thread Result[] reuse).
                    OffHeapTable table = new OffHeapTable();
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
                            processSegment(fileBase, chunkStart - fileBase, chunkEnd - fileBase, table);
                        }
                        catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                    threadResults[index] = table.extractRawAndFree();
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
        System.err.printf("Elapsed: %.3f s (threads=%d)%n", elapsed, workerThreadCount());
    }

    /**
     * Number of worker threads - matches CalculateAverage_thomaswue's own
     * Runtime.availableProcessors() (all cores, P+E on Apple Silicon), not
     * just the Performance cores. An earlier version of this file queried
     * `sysctl hw.perflevel0.physicalcpu` to stay P-core-only, reasoned from a
     * paired A/B that found using all 8 P-cores beat capping below that -
     * but that comparison never tested going past 8 into the 2 E-cores too.
     * A direct A/B here did: -DM1PRO.workers=10 (all cores) measured ~5.6%
     * faster wall-clock than the P-core-only default (2.07s vs 2.19s mean,
     * 5 rounds) despite doing *more* total CPU-seconds (17.63 vs 16.07) -
     * the E-cores did real, useful work rather than adding contention,
     * consistent with this file's segments being large and independent (no
     * shared mutable state to contend over), so a slower E-core thread just
     * finishes its own chunks later without blocking anyone else. Still
     * overridable with -DM1PRO.workers=N for further experimentation (e.g.
     * capping back to P-cores only on a workload where that assumption
     * doesn't hold).
     */
    private static int workerThreadCount() {
        return Integer.getInteger("M1PRO.workers", Runtime.getRuntime().availableProcessors());
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

    // v17: phase-level interleaving is now the default for width=4 (all 4
    // cursors' scans happen together, then all 4 hash/nameLen computations,
    // then all 4 number parses, then all 4 table.update() calls) instead of
    // row-level (each cursor's whole processRow - scan, hash, parse, update -
    // running to completion before the next cursor starts) - matching
    // CalculateAverage_thomaswue's own structure, which interleaves at the
    // phase level. A paired A/B (5 rounds, back-to-back) found phased ~3.5-
    // 3.7% faster on both wall-clock and CPU-seconds, not just smoother
    // scheduling - see processSegmentPhased4's Javadoc for the full
    // rationale, including why this lost when last tried (v7, before v8's
    // loop-free fixed-window scan existed) but doesn't lose for that same
    // reason any more. Overridable via -DM1PRO.interleaveMode=row to get the
    // old row-level path back for further A/B; only width=4 has a phased
    // variant, since this tests phase-vs-row structure, not distribution.
    private static final boolean INTERLEAVE_PHASED = !"row".equals(System.getProperty("M1PRO.interleaveMode"));

    private static void processSegment(long fileBase, long start, long end, OffHeapTable table) throws Exception {
        long base = fileBase + start;
        int limit = (int) (end - start);

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
            return;
        }

        if (INTERLEAVE_WIDTH == 4) {
            if (INTERLEAVE_PHASED) {
                processSegmentPhased4(base, limit, table);
            }
            else {
                processSegmentInterleaved4(base, limit, table);
            }
            return;
        }

        if (INTERLEAVE_WIDTH != 3) {
            processSegmentInterleavedN(base, limit, table);
            return;
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
    }

    /**
     * v12: hand-unrolled 4-way row-level interleaving, named locals (pos1..pos4),
     * matching the exact style of the 3-way block above - added specifically
     * because the general array-based N-way path below was found to carry its
     * own overhead (even at width=2, slower than the hand-unrolled 3-way
     * baseline), confounding any real signal about whether a wider width helps
     * on M1 Pro's core. This isolates the width variable from that
     * array-indexing/JIT-scheduling overhead. As of v17, this is no longer the
     * width=4 default - see processSegmentPhased4 below and INTERLEAVE_PHASED -
     * but stays available via -DM1PRO.interleaveMode=row for further A/B.
     */
    private static void processSegmentInterleaved4(long base, int limit, OffHeapTable table) {
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
    }

    /**
     * v17: phase-level interleaving, the default for width=4 (see
     * INTERLEAVE_PHASED; overridable to the row-level path above via
     * -DM1PRO.interleaveMode=row for further A/B). Row-level interleaving
     * runs each cursor's whole processRow (scan, hash, parse, table.update())
     * to completion before the next cursor starts; this instead groups the
     * same work by phase across all 4 cursors - all 4 scans, then all 4
     * hash/nameLen computations, then all 4 number parses, then all 4
     * table.update() calls - matching
     * CalculateAverage_thomaswue's own structure. The idea: every phase,
     * including the table update, has 3 other cursors' independent work in
     * flight to hide its latency, not just the scan phase.
     *
     * This was tried once before, in what's now v7 (see the class Javadoc's v8
     * entry), and lost. That test ran against the old variable-length SWAR
     * name-scanning loop - unrolling a loop with an unpredictable trip count 3-4
     * ways creates that many independent, unpredictable loop-back branches for
     * the scheduler to juggle, which was diagnosed at the time as the actual
     * reason phase-level interleaving lost, not phase-level grouping itself. v8
     * (already landed well before this) replaced that loop with the fixed
     * 16-byte-window scan processRow uses today - the same loop-free property
     * that makes thomaswue's own interleaving cheap in the first place. Phase-
     * level interleaving hasn't been re-tried since v8 landed, so this re-tests
     * it now that its stated reason for losing no longer applies.
     *
     * A cursor that can't use the fast 16-byte-window name scan, or whose name
     * is too close to the segment's end to safely fast-parse the number too,
     * falls out of the phased batch entirely for that iteration and completes
     * via the ordinary single-cursor processRow() - simplest correct handling
     * for what's a rare, segment-tail-only edge case, at the cost of a small
     * amount of redundant re-scanning for exactly those rows (processRow redoes
     * the name scan already done in phase 1) - not worth avoiding for how rarely
     * it triggers.
     */
    private static void processSegmentPhased4(long base, int limit, OffHeapTable table) {
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
            // Phase 1: all 4 cursors' fixed 16-byte-window name scans, matching
            // processRow's own fast path. A cursor that can't use it (delimiter
            // not found in the window, too close to limit, or the number after
            // it too close to limit to fast-parse) falls back to a plain
            // processRow() call for its row right here and is done for this
            // iteration - phases 2-4 skip it.
            boolean fast1 = false;
            int nameStart1 = pos1;
            int nameLen1 = 0, afterDelimiter1 = 0;
            long word1 = 0, word1b = 0;
            if (pos1 + 16 <= limit) {
                word1 = UNSAFE.getLong(base + pos1);
                word1b = UNSAFE.getLong(base + pos1 + 8);
                long delimiterMask1 = findDelimiter(word1);
                long delimiterMask1b = findDelimiter(word1b);
                if ((delimiterMask1 | delimiterMask1b) != 0) {
                    int letterCount1 = Long.numberOfTrailingZeros(delimiterMask1) >>> 3;
                    int letterCount1b = Long.numberOfTrailingZeros(delimiterMask1b) >>> 3;
                    long sel1 = FAST_MASK2[letterCount1];
                    nameLen1 = (int) (letterCount1 + (letterCount1b & sel1));
                    afterDelimiter1 = nameStart1 + nameLen1 + 1;
                    fast1 = afterDelimiter1 + 8 <= limit;
                }
            }
            if (!fast1) {
                pos1 = processRow(base, limit, pos1, table);
            }

            boolean fast2 = false;
            int nameStart2 = pos2;
            int nameLen2 = 0, afterDelimiter2 = 0;
            long word2 = 0, word2b = 0;
            if (pos2 + 16 <= limit) {
                word2 = UNSAFE.getLong(base + pos2);
                word2b = UNSAFE.getLong(base + pos2 + 8);
                long delimiterMask2 = findDelimiter(word2);
                long delimiterMask2b = findDelimiter(word2b);
                if ((delimiterMask2 | delimiterMask2b) != 0) {
                    int letterCount2 = Long.numberOfTrailingZeros(delimiterMask2) >>> 3;
                    int letterCount2b = Long.numberOfTrailingZeros(delimiterMask2b) >>> 3;
                    long sel2 = FAST_MASK2[letterCount2];
                    nameLen2 = (int) (letterCount2 + (letterCount2b & sel2));
                    afterDelimiter2 = nameStart2 + nameLen2 + 1;
                    fast2 = afterDelimiter2 + 8 <= limit;
                }
            }
            if (!fast2) {
                pos2 = processRow(base, limit, pos2, table);
            }

            boolean fast3 = false;
            int nameStart3 = pos3;
            int nameLen3 = 0, afterDelimiter3 = 0;
            long word3 = 0, word3b = 0;
            if (pos3 + 16 <= limit) {
                word3 = UNSAFE.getLong(base + pos3);
                word3b = UNSAFE.getLong(base + pos3 + 8);
                long delimiterMask3 = findDelimiter(word3);
                long delimiterMask3b = findDelimiter(word3b);
                if ((delimiterMask3 | delimiterMask3b) != 0) {
                    int letterCount3 = Long.numberOfTrailingZeros(delimiterMask3) >>> 3;
                    int letterCount3b = Long.numberOfTrailingZeros(delimiterMask3b) >>> 3;
                    long sel3 = FAST_MASK2[letterCount3];
                    nameLen3 = (int) (letterCount3 + (letterCount3b & sel3));
                    afterDelimiter3 = nameStart3 + nameLen3 + 1;
                    fast3 = afterDelimiter3 + 8 <= limit;
                }
            }
            if (!fast3) {
                pos3 = processRow(base, limit, pos3, table);
            }

            boolean fast4 = false;
            int nameStart4 = pos4;
            int nameLen4 = 0, afterDelimiter4 = 0;
            long word4 = 0, word4b = 0;
            if (pos4 + 16 <= limit) {
                word4 = UNSAFE.getLong(base + pos4);
                word4b = UNSAFE.getLong(base + pos4 + 8);
                long delimiterMask4 = findDelimiter(word4);
                long delimiterMask4b = findDelimiter(word4b);
                if ((delimiterMask4 | delimiterMask4b) != 0) {
                    int letterCount4 = Long.numberOfTrailingZeros(delimiterMask4) >>> 3;
                    int letterCount4b = Long.numberOfTrailingZeros(delimiterMask4b) >>> 3;
                    long sel4 = FAST_MASK2[letterCount4];
                    nameLen4 = (int) (letterCount4 + (letterCount4b & sel4));
                    afterDelimiter4 = nameStart4 + nameLen4 + 1;
                    fast4 = afterDelimiter4 + 8 <= limit;
                }
            }
            if (!fast4) {
                pos4 = processRow(base, limit, pos4, table);
            }

            // Phase 2: all 4 cursors' name-word masking + hash, matching
            // finishRow's own masking logic - only for cursors still in the
            // phased batch (fast1..fast4).
            long firstWord1 = 0, secondWord1 = 0;
            int hash1 = 0;
            if (fast1) {
                if (nameLen1 <= 8) {
                    firstWord1 = word1 & MASK1[nameLen1 - 1];
                }
                else if (nameLen1 < 16) {
                    firstWord1 = word1;
                    secondWord1 = word1b & MASK1[nameLen1 - 9];
                }
                else {
                    firstWord1 = word1;
                    secondWord1 = word1b;
                }
                long combined1 = firstWord1 ^ secondWord1;
                hash1 = finishName((int) combined1 ^ (int) (combined1 >>> 32));
            }

            long firstWord2 = 0, secondWord2 = 0;
            int hash2 = 0;
            if (fast2) {
                if (nameLen2 <= 8) {
                    firstWord2 = word2 & MASK1[nameLen2 - 1];
                }
                else if (nameLen2 < 16) {
                    firstWord2 = word2;
                    secondWord2 = word2b & MASK1[nameLen2 - 9];
                }
                else {
                    firstWord2 = word2;
                    secondWord2 = word2b;
                }
                long combined2 = firstWord2 ^ secondWord2;
                hash2 = finishName((int) combined2 ^ (int) (combined2 >>> 32));
            }

            long firstWord3 = 0, secondWord3 = 0;
            int hash3 = 0;
            if (fast3) {
                if (nameLen3 <= 8) {
                    firstWord3 = word3 & MASK1[nameLen3 - 1];
                }
                else if (nameLen3 < 16) {
                    firstWord3 = word3;
                    secondWord3 = word3b & MASK1[nameLen3 - 9];
                }
                else {
                    firstWord3 = word3;
                    secondWord3 = word3b;
                }
                long combined3 = firstWord3 ^ secondWord3;
                hash3 = finishName((int) combined3 ^ (int) (combined3 >>> 32));
            }

            long firstWord4 = 0, secondWord4 = 0;
            int hash4 = 0;
            if (fast4) {
                if (nameLen4 <= 8) {
                    firstWord4 = word4 & MASK1[nameLen4 - 1];
                }
                else if (nameLen4 < 16) {
                    firstWord4 = word4;
                    secondWord4 = word4b & MASK1[nameLen4 - 9];
                }
                else {
                    firstWord4 = word4;
                    secondWord4 = word4b;
                }
                long combined4 = firstWord4 ^ secondWord4;
                hash4 = finishName((int) combined4 ^ (int) (combined4 >>> 32));
            }

            // Phase 3: all 4 cursors' branchless number parses, matching
            // finishRow's own fast path.
            int value1 = 0, newPos1 = afterDelimiter1;
            if (fast1) {
                long numberWord1 = UNSAFE.getLong(base + afterDelimiter1);
                int decimalSepPos1 = Long.numberOfTrailingZeros(~numberWord1 & 0x10101000L);
                value1 = (int) convertIntoNumber(decimalSepPos1, numberWord1);
                newPos1 = afterDelimiter1 + (decimalSepPos1 >>> 3) + 3;
            }

            int value2 = 0, newPos2 = afterDelimiter2;
            if (fast2) {
                long numberWord2 = UNSAFE.getLong(base + afterDelimiter2);
                int decimalSepPos2 = Long.numberOfTrailingZeros(~numberWord2 & 0x10101000L);
                value2 = (int) convertIntoNumber(decimalSepPos2, numberWord2);
                newPos2 = afterDelimiter2 + (decimalSepPos2 >>> 3) + 3;
            }

            int value3 = 0, newPos3 = afterDelimiter3;
            if (fast3) {
                long numberWord3 = UNSAFE.getLong(base + afterDelimiter3);
                int decimalSepPos3 = Long.numberOfTrailingZeros(~numberWord3 & 0x10101000L);
                value3 = (int) convertIntoNumber(decimalSepPos3, numberWord3);
                newPos3 = afterDelimiter3 + (decimalSepPos3 >>> 3) + 3;
            }

            int value4 = 0, newPos4 = afterDelimiter4;
            if (fast4) {
                long numberWord4 = UNSAFE.getLong(base + afterDelimiter4);
                int decimalSepPos4 = Long.numberOfTrailingZeros(~numberWord4 & 0x10101000L);
                value4 = (int) convertIntoNumber(decimalSepPos4, numberWord4);
                newPos4 = afterDelimiter4 + (decimalSepPos4 >>> 3) + 3;
            }

            // Phase 4: all 4 cursors' table.update() calls, last, as their own
            // batch.
            if (fast1) {
                table.update(base + nameStart1, nameLen1, hash1, firstWord1, secondWord1, value1);
                pos1 = newPos1;
            }
            if (fast2) {
                table.update(base + nameStart2, nameLen2, hash2, firstWord2, secondWord2, value2);
                pos2 = newPos2;
            }
            if (fast3) {
                table.update(base + nameStart3, nameLen3, hash3, firstWord3, secondWord3, value3);
                pos3 = newPos3;
            }
            if (fast4) {
                table.update(base + nameStart4, nameLen4, hash4, firstWord4, secondWord4, value4);
                pos4 = newPos4;
            }
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
    }

    /**
     * v12: general N-way interleaving, for the -DM1PRO.interleaveWidth=N sweep -
     * see the field Javadoc. Splits the segment into N roughly equal, row-aligned
     * parts and round-robins processRow() across all N cursors per outer iteration,
     * same principle as the hand-unrolled 3-way version above, just array-indexed
     * to support a runtime-chosen width instead of 3 named locals.
     */
    private static void processSegmentInterleavedN(long base, int limit, OffHeapTable table) {
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
                    return;
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

        table.update(base + nameStart, nameLen, hash, firstWord, secondWord, value);
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
     * One occupied slot's data, extracted from a thread's off-heap table
     * with no String decoding - just the raw (address,len) the name lives
     * at plus its aggregates. addr is an absolute address into the one
     * whole-file mapping (see main()'s v14 note; v16 widened this from a
     * separate base+offset pair once OffHeapTable itself went absolute -
     * see its own Javadoc), kept alive for the JVM's lifetime by
     * Arena.global() - no per-segment reachability tracking needed.
     */
    private static final class RawEntry {
        final long addr;
        final int len;
        final int hash;
        final long sum;
        final int count;
        final short min, max;

        RawEntry(long addr, int len, int hash, long sum, int count, short min, short max) {
            this.addr = addr;
            this.len = len;
            this.hash = hash;
            this.sum = sum;
            this.count = count;
            this.min = min;
            this.max = max;
        }
    }

    /**
     * Off-heap, fixed-stride open-addressing hash table. One persistent
     * instance per worker thread (v16; see class Javadoc), reused across
     * every chunk that thread claims over its whole run; keys are zero-copy
     * absolute-address references into the one whole-file mapping, so
     * equality checks compare raw bytes straight out of the mapped file
     * with no intermediate allocation.
     */
    private static final class OffHeapTable {
        private int capacity = 1 << 14; // 16384 slots * 32B = 512KB
        private int mask = capacity - 1;
        private long table = UNSAFE.allocateMemory((long) capacity * SLOT_SIZE);
        private int size = 0;

        // v16: no more per-table segmentBase - see class Javadoc. One table
        // is now created per worker thread (not per chunk) and reused across
        // every chunk that thread claims off the AtomicLong cursor over its
        // whole run, so there's no single chunk base it could meaningfully
        // hold any more; update() takes an absolute nameAddr instead.
        OffHeapTable() {
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
        void update(long nameAddr, int nameLen, int hash, long firstWord, long secondWord, int value) {
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
                    UNSAFE.putLong(addr + 16, nameAddr);
                    UNSAFE.putShort(addr + 24, (short) nameLen);
                    UNSAFE.putInt(addr + 28, hash);
                    UNSAFE.putLong(addr + 32, firstWord);
                    UNSAFE.putLong(addr + 40, secondWord);
                    size++;
                    if (size * 2 > capacity)
                        resize();
                    return;
                }

                if (storedFirst == firstWord && storedSecond == secondWord
                        && (!needsFullCompare || regionEquals(UNSAFE.getLong(addr + 16), nameAddr, nameLen))) {
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
                short len = UNSAFE.getShort(addr + 24);
                if (len != 0) {
                    int hash = UNSAFE.getInt(addr + 28);
                    int idx = hash & newMask;
                    while (true) {
                        long newAddr = slotAddr(newTable, idx);
                        if (UNSAFE.getShort(newAddr + 24) == 0) {
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
         * memory - that's still referenced (by absolute address) from the
         * returned entries, and stays valid for the JVM's lifetime via
         * Arena.global() (see main()'s v14 note). As of v16, called once per
         * worker thread when its cursor claims are exhausted, not once per
         * chunk - see class Javadoc.
         */
        List<RawEntry> extractRawAndFree() {
            List<RawEntry> list = new ArrayList<>(size);
            for (int i = 0; i < capacity; i++) {
                long addr = slotAddr(table, i);
                short len = UNSAFE.getShort(addr + 24);
                if (len != 0) {
                    long nameAddr = UNSAFE.getLong(addr + 16);
                    int hash = UNSAFE.getInt(addr + 28);
                    long sum = UNSAFE.getLong(addr);
                    int count = UNSAFE.getInt(addr + 8);
                    short min = UNSAFE.getShort(addr + 12);
                    short max = UNSAFE.getShort(addr + 14);
                    list.add(new RawEntry(nameAddr, len, hash, sum, count, min, max));
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
        private long[] keyAddr = new long[capacity];
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
                    keyAddr[idx] = e.addr;
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
                if (len == e.len && regionEquals(keyAddr[idx], e.addr, e.len)) {
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
            long[] nKeyAddr = new long[newCapacity];
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
                    nKeyAddr[idx] = keyAddr[i];
                    nKeyLen[idx] = keyLen[i];
                    nHashes[idx] = hashes[i];
                    nSums[idx] = sums[i];
                    nCounts[idx] = counts[i];
                    nMins[idx] = mins[i];
                    nMaxs[idx] = maxs[i];
                }
            }
            keyAddr = nKeyAddr;
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
                    long addr = keyAddr[i];
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
