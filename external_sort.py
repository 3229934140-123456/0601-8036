#!/usr/bin/env python3
"""
外部排序程序：在4GB内存机器上排序40GB数据文件
- 阶段一：分块内排序生成有序临时文件
- 阶段二：K路归并输出最终有序文件

核心设计问题：最优扇入路数 K 的选择
"""

import os
import sys
import heapq
import math
import tempfile
import argparse
import struct
import time
import array
from typing import List, BinaryIO, Generator, Tuple


# ============================================================
# 配置参数
# ============================================================
DEFAULT_MEMORY_MB = 4000        # 可用内存（MB），预留部分给系统和Python开销
DEFAULT_BLOCK_SIZE = 4096       # 单次I/O块大小（字节）
RECORD_SIZE = 8                 # 每条记录大小（8字节，采用64位整数）
TEMP_DIR_PREFIX = "ext_sort_"
# 分块内排序安全系数：Python int 对象每条约 32-40 字节（8 字节数据 + PyLongObject 开销）
# 数据 8B → 对象 40B ≈ 5 倍膨胀，Timsort 最坏 50% 临时空间
# 总安全系数 = 5 * 1.5 = 7.5，取 8 留余量
PYINT_SORT_OVERHEAD = 8.0


# ============================================================
# 参数校验与标准化
# ============================================================
def validate_and_normalize_params(memory_mb: int, block_size: int) -> Tuple[int, int]:
    """
    校验并规范化输入参数：
    - memory_mb 必须为正整数
    - block_size 必须 >= RECORD_SIZE 且是 RECORD_SIZE 的倍数
      否则自动向上对齐并给出警告
    
    返回: (memory_mb, normalized_block_size)
    """
    if memory_mb <= 0:
        raise ValueError(
            f"参数错误：可用内存 memory_mb={memory_mb} 必须为正整数（MB）"
        )
    
    if block_size <= 0:
        raise ValueError(
            f"参数错误：I/O 块大小 block_size={block_size} 必须为正整数（字节）"
        )
    
    if block_size < RECORD_SIZE:
        aligned = RECORD_SIZE
        print(f"[警告] I/O 块大小 {block_size}B < 记录大小 {RECORD_SIZE}B，"
              f"已自动对齐到 {aligned}B")
        block_size = aligned
    elif block_size % RECORD_SIZE != 0:
        aligned = ((block_size + RECORD_SIZE - 1) // RECORD_SIZE) * RECORD_SIZE
        print(f"[警告] I/O 块大小 {block_size}B 不是记录大小 {RECORD_SIZE}B 的倍数，"
              f"已自动向上对齐到 {aligned}B")
        block_size = aligned
    
    return memory_mb, block_size


# ============================================================
# 阶段一：分块内排序
# ============================================================
def compute_chunk_size(memory_mb: int, block_size: int) -> int:
    """
    计算每个数据块的原始字节数（list+int对象安全上限）
    
    由于 Python int 对象膨胀：8 字节数据 → 约 40 字节对象（5 倍膨胀）
    Timsort 最坏需要 50% 临时空间
    因此原始数据大小上限 = memory / (5 * 1.5) ≈ memory / 7.5，取 8 作安全系数
    
    公式:
        raw_chunk_bytes = (memory_mb * 1024 * 1024 / PYINT_SORT_OVERHEAD)
        再向下取整到 block_size 倍数
    """
    max_raw_bytes = int(memory_mb * 1024 * 1024 / PYINT_SORT_OVERHEAD)
    chunk_bytes = (max_raw_bytes // block_size) * block_size
    return max(chunk_bytes, block_size)


def split_and_sort(input_file: str, temp_dir: str,
                   memory_mb: int, block_size: int) -> List[str]:
    """
    阶段一：分块读取 -> 内存排序 -> 写入临时文件
    安全系数已考虑 Python int 对象 5 倍膨胀 + Timsort 50% 临时空间
    
    返回有序临时文件路径列表（空文件时返回空列表）
    """
    chunk_bytes = compute_chunk_size(memory_mb, block_size)
    temp_files: List[str] = []
    records_per_chunk = chunk_bytes // RECORD_SIZE
    
    file_size = os.path.getsize(input_file)
    print(f"[阶段一] 分块大小: {chunk_bytes / (1024*1024):.2f} MB, "
          f"每块约 {records_per_chunk:,} 条记录（已预留 {PYINT_SORT_OVERHEAD:.0f}x 安全系数）")
    
    # 空输入文件：直接返回空列表，由调用方创建空结果文件
    if file_size == 0:
        print("[阶段一] 输入文件为空，跳过分块排序")
        return temp_files
    
    # 检查输入文件大小是否是记录大小的整数倍
    if file_size % RECORD_SIZE != 0:
        print(f"[警告] 输入文件大小 {file_size}B 不是记录大小 {RECORD_SIZE}B 的倍数，"
              f"末尾的 {file_size % RECORD_SIZE} 字节将被忽略")
    
    total_chunks = math.ceil(file_size / chunk_bytes)
    
    with open(input_file, 'rb') as fin:
        chunk_idx = 0
        while True:
            raw_data = fin.read(chunk_bytes)
            if not raw_data:
                break
            
            num_records = len(raw_data) // RECORD_SIZE
            if num_records == 0:
                break
            
            # 使用 struct.unpack + list（list.sort() 原地排序，内存稳定）
            # 注意：Python int 对象有 ~5 倍内存膨胀，已在 compute_chunk_size 中预留
            records = list(struct.unpack(f'<{num_records}q', raw_data[:num_records * RECORD_SIZE]))
            
            # 原地排序（Timsort，稳定且内存行为可预测）
            records.sort()
            
            temp_path = os.path.join(temp_dir, f"sorted_{chunk_idx:06d}.tmp")
            with open(temp_path, 'wb') as fout:
                fout.write(struct.pack(f'<{len(records)}q', *records))
            
            # 释放引用，帮助 GC
            del records
            
            temp_files.append(temp_path)
            chunk_idx += 1
            print(f"  已生成临时块 {chunk_idx}/{total_chunks}: {temp_path}")
    
    print(f"[阶段一完成] 共生成 {len(temp_files)} 个有序临时文件")
    return temp_files


# ============================================================
# 最优扇入路数计算（核心理论部分）
# ============================================================
def compute_optimal_fanin(memory_mb: int, block_size: int, num_runs: int) -> int:
    """
    计算归并阶段的最优扇入路数 K
    
    【数学关系推导】
    
    设：
    - M = 可用内存大小（字节）
    - B = 单次 I/O 块大小（字节）
    - K = 扇入路数（同时归并的有序文件数）
    
    在 K 路归并中，每路至少需要 1 个输出缓冲区 + K 个输入缓冲区，
    每个缓冲区大小为 B：
        (K + 1) * B ≤ M
    
    每路缓冲区的平均大小为：
        buf_per_run = M / (K + 1)
    
    归并阶段发生的 I/O 次数取决于两个因素：
    1. **顺序 I/O 开销**：数据必须被读入和写出，这部分与 K 无关，是固定的
       总数据量 D 的顺序读写 = 2 * D / B 次 I/O
       
    2. **寻道/切换开销**：当每路缓冲区变小时，需要更频繁地切换输入文件，
       每路需要读取的块数 = (D/K) / buf_per_run = (D/K) / (M/(K+1)) ≈ D*K/(K*M) = D/M
       这看似与 K 无关，但考虑到每读一个块都要打开/切换文件流，
       以及输出缓冲区的写回频率，实际 I/O 次数模型为：
       
       IO(K) = (K + 1) * (D / M)  ...  当缓冲区足够大时
       
       但当 K 过大时，每路缓冲区 buf_per_run < B，此时必须将缓冲区设为 B，
       约束 (K+1)*B ≤ M 无法满足，会导致**颠簸（thrashing）**：
       
       此时每路只能读 B 字节，需要 K+1 次 I/O 才能处理 (K+1)*B 数据，
       而每次处理的数据量只有 K*B，I/O 利用率 = K/(K+1) → 趋近于 1 但绝对次数增加
    
    【最优解】
    
    实际工程中，最优扇入满足：
        K_opt = floor( M / B ) - 1
    
    证明：设归并总轮数为 R = ceil(log_K N)，其中 N 是临时文件数
         总 I/O = R * D（读） + R * D（写） = 2D * R
         要最小化 R，需要最大化 K
         但 K 受约束 (K+1)*B ≤ M → K ≤ M/B - 1
    
    因此取 K_opt = floor(M / B) - 1
    """
    # 空输入场景：不需要归并
    if num_runs <= 1:
        print(f"[扇入计算] 临时文件数={num_runs}，无需归并")
        return 1
    
    M = memory_mb * 1024 * 1024  # 转为字节
    K_opt = (M // block_size) - 1  # -1 是留给输出缓冲区
    
    # 实际还要考虑：如果临时文件数 N < K_opt，那么 K 取 N 即可
    # 另外，实际工程中会留一些余量，取 K_opt 的 70%~80% 以应对文件句柄等开销
    K_practical = max(2, min(num_runs, int(K_opt * 0.75)))
    
    print(f"[扇入计算] 理论最优 K = floor({memory_mb}MB / {block_size}B) - 1 = {K_opt:,}")
    print(f"[扇入计算] 实际取 K = {K_practical:,}（留 25% 余量，不超过临时文件数 {num_runs}）")
    
    return K_practical


def compute_io_count_model(memory_mb: int, block_size: int, 
                           num_runs: int, K: int) -> dict:
    """
    计算不同 K 值下的预估 I/O 次数模型，用于解释"扇入并非越大越好"
    
    返回包含理论分析的字典
    """
    M = memory_mb * 1024 * 1024
    
    # 每路输入缓冲区大小
    buf_per_input = M // (K + 1)  # 预留 1 个输出缓冲区
    
    # 当 buf_per_input < block_size 时，发生颠簸
    thrashing = buf_per_input < block_size
    
    # 归并轮数 R = ceil(log_K num_runs)
    if K >= num_runs:
        R = 1
    else:
        R = math.ceil(math.log(num_runs, K))
    
    return {
        'K': K,
        'buf_per_input_MB': buf_per_input / (1024*1024),
        'thrashing': thrashing,
        'merge_rounds': R,
        'constraint_ok': (K + 1) * block_size <= M
    }


# ============================================================
# 阶段二：多路归并
# ============================================================
class RunReader:
    """
    有序运行文件的读取器，带缓冲
    使用 array.frombytes 紧凑存储，避免 int 对象膨胀
    """
    def __init__(self, filepath: str, block_size: int):
        self.filepath = filepath
        self.block_size = block_size
        self.fp = open(filepath, 'rb')
        self.buffer: array.ArrayType[int] = array.array('q')
        self.buffer_idx = 0
        self.file_done = False
        self._refill()
    
    def _refill(self):
        """从磁盘以块为单位填充缓冲区（紧凑数组）"""
        raw = self.fp.read(self.block_size)
        if not raw:
            self.file_done = True
            self.buffer = array.array('q')
            self.buffer_idx = 0
            return
        self.buffer = array.array('q')
        self.buffer.frombytes(raw)
        self.buffer_idx = 0
    
    def peek(self):
        """查看当前头部元素但不弹出，无元素返回 None"""
        while self.buffer_idx >= len(self.buffer) and not self.file_done:
            self._refill()
        if self.buffer_idx < len(self.buffer):
            return self.buffer[self.buffer_idx]
        return None
    
    def pop(self):
        """弹出当前头部元素"""
        val = self.peek()
        if val is not None:
            self.buffer_idx += 1
        return val
    
    def close(self):
        self.fp.close()


class BufferedWriter:
    """
    带缓冲的输出写入器，使用紧凑数组缓冲
    """
    def __init__(self, filepath: str, block_size: int):
        self.filepath = filepath
        self.block_size = block_size
        self.fp = open(filepath, 'wb')
        self.buffer: array.ArrayType[int] = array.array('q')
        # 缓冲区记录数阈值
        self._threshold = block_size // RECORD_SIZE
    
    def write(self, value: int):
        self.buffer.append(value)
        if len(self.buffer) >= self._threshold:
            self._flush()
    
    def _flush(self):
        if self.buffer:
            self.buffer.tofile(self.fp)
            self.buffer = array.array('q')
    
    def close(self):
        self._flush()
        self.fp.close()


def k_way_merge(run_files: List[str], output_file: str,
                K: int, block_size: int, temp_dir: str) -> str:
    """
    执行 K 路归并（可能需要多轮）
    
    参数:
        run_files: 待归并的有序文件列表（空列表时直接创建空输出文件）
        output_file: 最终输出文件路径
        K: 扇入路数
        block_size: I/O 块大小
        temp_dir: 临时目录
    
    返回: 最终有序文件路径
    """
    # 空输入：直接创建空输出文件
    if not run_files:
        print("  [归并] 无输入临时文件，直接创建空输出文件")
        with open(output_file, 'wb') as f:
            pass
        print(f"[阶段二完成] 归并结果写入: {output_file}（空文件）")
        return output_file
    
    # 只有一个临时文件：直接拷贝到输出
    if len(run_files) == 1:
        print(f"  [归并] 仅 1 个临时文件，直接复制到输出")
        import shutil
        if os.path.exists(output_file):
            os.remove(output_file)
        shutil.copy2(run_files[0], output_file)
        print(f"[阶段二完成] 归并结果写入: {output_file}")
        return output_file
    
    current_runs = run_files.copy()
    round_idx = 1
    
    while len(current_runs) > 1:
        actual_K = min(K, len(current_runs))
        print(f"  [归并第 {round_idx} 轮] {len(current_runs)} → "
              f"{math.ceil(len(current_runs) / actual_K)} 路，扇入 K={actual_K}")
        
        next_runs: List[str] = []
        is_final_round = (len(current_runs) <= actual_K)
        
        # 按 K 个一组进行归并
        for group_start in range(0, len(current_runs), actual_K):
            group = current_runs[group_start: group_start + actual_K]
            
            # 最后一组决定输出位置
            if is_final_round and group_start + len(group) >= len(current_runs):
                out_path = output_file
            else:
                out_path = os.path.join(temp_dir, f"merge_{round_idx}_{len(next_runs):06d}.tmp")
            
            # 单组归并
            _merge_single_group(group, out_path, block_size)
            next_runs.append(out_path)
        
        # 清理上一轮的临时文件（可选，节省磁盘空间）
        for f in current_runs:
            if f not in next_runs and f != output_file:
                try:
                    os.remove(f)
                except OSError:
                    pass
        
        current_runs = next_runs
        round_idx += 1
    
    print(f"[阶段二完成] 归并结果写入: {output_file}")
    return output_file


def _merge_single_group(input_files: List[str], output_file: str, block_size: int):
    """
    使用最小堆完成一组（≤K 路）归并
    """
    readers: List[RunReader] = [RunReader(f, block_size) for f in input_files]
    writer = BufferedWriter(output_file, block_size)
    
    try:
        # 初始化堆：(value, reader_index)
        heap: List[Tuple[int, int]] = []
        for idx, reader in enumerate(readers):
            val = reader.peek()
            if val is not None:
                heapq.heappush(heap, (val, idx))
        
        while heap:
            min_val, min_idx = heapq.heappop(heap)
            writer.write(min_val)
            readers[min_idx].pop()  # 移除已写出的元素
            
            # 补充该路的下一个元素
            next_val = readers[min_idx].peek()
            if next_val is not None:
                heapq.heappush(heap, (next_val, min_idx))
    finally:
        for r in readers:
            r.close()
        writer.close()


# ============================================================
# 主入口
# ============================================================
def external_sort(input_file: str, output_file: str,
                  memory_mb: int = DEFAULT_MEMORY_MB,
                  block_size: int = DEFAULT_BLOCK_SIZE,
                  keep_temp: bool = False) -> None:
    """
    完整外部排序流程
    
    参数:
        input_file: 待排序输入文件（每条记录为 8 字节小端整数）
        output_file: 排序后输出文件
        memory_mb: 可用内存大小（MB）
        block_size: 单次 I/O 块大小（字节）
        keep_temp: 是否保留临时文件（用于调试）
    """
    # ============== 参数校验与标准化 ==============
    memory_mb, block_size = validate_and_normalize_params(memory_mb, block_size)
    
    start_time = time.time()
    phase1_time = 0.0
    phase2_time = 0.0
    
    # 检查输入文件
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    file_size = os.path.getsize(input_file)
    file_size_gb = file_size / (1024**3)
    print("=" * 60)
    print(f"外部排序启动")
    print(f"  输入文件: {input_file} ({file_size_gb:.2f} GB, {file_size:,} 字节)")
    print(f"  输出文件: {output_file}")
    print(f"  可用内存: {memory_mb} MB")
    print(f"  I/O 块大小: {block_size} B")
    print("=" * 60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    print(f"临时目录: {temp_dir}\n")
    
    try:
        # ========== 阶段一：分块排序 ==========
        print("--- 阶段一：分块内排序 ---")
        phase1_start = time.time()
        sorted_runs = split_and_sort(input_file, temp_dir, memory_mb, block_size)
        phase1_time = time.time() - phase1_start
        
        # ========== 计算最优扇入 ==========
        print("\n--- 最优扇入计算 ---")
        num_runs = len(sorted_runs)
        K_opt = compute_optimal_fanin(memory_mb, block_size, num_runs)
        
        # 打印对比分析（有多个临时文件时才有意义）
        if num_runs >= 2:
            print("\n  K 值敏感性分析（解释：扇入并非越大越好）：")
            K_candidates = [2, 8, 32, 128, 512, 2048, 8192, K_opt]
            if K_opt * 1.5 > K_opt:
                K_candidates += [int(K_opt * 1.5), int(K_opt * 3)]
            seen = set()
            for K_test in K_candidates:
                if K_test < 2 or K_test in seen:
                    continue
                seen.add(K_test)
                info = compute_io_count_model(memory_mb, block_size, num_runs, K_test)
                status = []
                if not info['constraint_ok']:
                    status.append("⚠️ 违反内存约束")
                if info['thrashing']:
                    status.append("❌ 缓冲区<块大小，发生颠簸!")
                status_str = " | ".join(status) if status else "✅ 正常"
                print(f"    K={K_test:>6,} | 每路缓冲={info['buf_per_input_MB']:>7.2f}MB "
                      f"| 归并轮数={info['merge_rounds']} | {status_str}")
        
        # ========== 阶段二：多路归并 ==========
        print("\n--- 阶段二：多路归并 ---")
        phase2_start = time.time()
        k_way_merge(sorted_runs, output_file, K_opt, block_size, temp_dir)
        phase2_time = time.time() - phase2_start
        
    finally:
        # 清理临时文件
        if not keep_temp and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"\n临时目录已清理: {temp_dir}")
            except OSError as e:
                print(f"\n警告：清理临时目录失败: {e}")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"排序完成！")
    print(f"  输出文件: {output_file} ({os.path.getsize(output_file):,} 字节)")
    print(f"  阶段一耗时: {phase1_time:.1f}s")
    print(f"  阶段二耗时: {phase2_time:.1f}s")
    print(f"  总计耗时: {total_time:.1f}s")
    print("=" * 60)


# ============================================================
# 辅助工具：生成测试数据
# ============================================================
def generate_test_data(output_file: str, num_records: int) -> None:
    """
    生成随机测试数据（8字节小端整数）
    
    参数:
        output_file: 输出文件路径
        num_records: 记录条数，0 时生成空文件
    """
    import random
    file_size_gb = (num_records * RECORD_SIZE) / (1024**3)
    print(f"生成测试数据: {num_records:,} 条记录, 约 {file_size_gb:.2f} GB")
    
    # 空文件：直接创建
    if num_records == 0:
        with open(output_file, 'wb') as f:
            pass
        print(f"测试数据生成完成: {output_file}（空文件）")
        return
    
    batch_size = 1_000_000  # 每批 100 万条（约 8MB）
    written = 0
    
    with open(output_file, 'wb') as f:
        while written < num_records:
            n = min(batch_size, num_records - written)
            records = array.array('q', (random.randint(-(1<<63), (1<<63)-1) for _ in range(n)))
            records.tofile(f)
            written += n
            print(f"  已写入 {written:,}/{num_records:,} 条 ({written/num_records*100:.1f}%)")
    
    print(f"测试数据生成完成: {output_file}")


def verify_sorted(filepath: str) -> bool:
    """
    验证文件是否有序（升序）
    - 空文件视为有序
    - 非空文件按记录逐条比较
    
    返回: True=有序, False=无序
    """
    print(f"验证排序结果: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ❌ 文件不存在: {filepath}")
        return False
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        print(f"  ✅ 空文件，天然有序（0 条记录）")
        return True
    
    if file_size % RECORD_SIZE != 0:
        print(f"  ⚠️  文件大小 {file_size}B 不是记录大小 {RECORD_SIZE}B 的倍数，"
              f"末尾 {file_size % RECORD_SIZE}B 将被忽略")
    
    block_records = 100_000
    prev_val = None
    record_count = 0
    
    with open(filepath, 'rb') as f:
        while True:
            raw = f.read(block_records * RECORD_SIZE)
            if not raw:
                break
            records = array.array('q')
            records.frombytes(raw)
            
            for val in records:
                if prev_val is not None and val < prev_val:
                    print(f"  ❌ 排序错误！位置 {record_count}: "
                          f"{prev_val} > {val}")
                    return False
                prev_val = val
                record_count += 1
            
            if record_count % 1_000_000 == 0:
                print(f"  已验证 {record_count:,} 条...")
    
    print(f"  ✅ 文件有序，共 {record_count:,} 条记录")
    return True


# ============================================================
# 命令行接口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="外部排序程序：在有限内存下排序超大文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 5GB 测试数据
  python external_sort.py generate -o test.dat -n 671088640
  
  # 执行排序（默认 4GB 内存，4KB 块大小）
  python external_sort.py sort -i test.dat -o sorted.dat
  
  # 使用 2GB 内存，8KB 块大小
  python external_sort.py sort -i test.dat -o sorted.dat -m 2000 -b 8192
  
  # 验证排序结果（成功退出码 0，失败退出码 1）
  python external_sort.py verify -f sorted.dat
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # generate 子命令
    gen_parser = subparsers.add_parser("generate", help="生成测试数据")
    gen_parser.add_argument("-o", "--output", required=True, help="输出文件路径")
    gen_parser.add_argument("-n", "--num-records", type=int, required=True,
                            help="记录条数（每条约 8 字节），0 时生成空文件")
    
    # sort 子命令
    sort_parser = subparsers.add_parser("sort", help="执行外部排序")
    sort_parser.add_argument("-i", "--input", required=True, help="待排序输入文件")
    sort_parser.add_argument("-o", "--output", required=True, help="排序后输出文件")
    sort_parser.add_argument("-m", "--memory-mb", type=int, default=DEFAULT_MEMORY_MB,
                             help=f"可用内存大小(MB)，默认 {DEFAULT_MEMORY_MB}")
    sort_parser.add_argument("-b", "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                             help=f"I/O块大小(字节)，默认 {DEFAULT_BLOCK_SIZE}")
    sort_parser.add_argument("--keep-temp", action="store_true", help="保留临时文件")
    
    # verify 子命令
    ver_parser = subparsers.add_parser("verify", help="验证文件是否有序")
    ver_parser.add_argument("-f", "--file", required=True, help="待验证文件路径")
    
    args = parser.parse_args()
    
    try:
        if args.command == "generate":
            if args.num_records < 0:
                print(f"错误：记录条数 num_records 不能为负", file=sys.stderr)
                sys.exit(2)
            generate_test_data(args.output, args.num_records)
            sys.exit(0)
            
        elif args.command == "sort":
            external_sort(
                input_file=args.input,
                output_file=args.output,
                memory_mb=args.memory_mb,
                block_size=args.block_size,
                keep_temp=args.keep_temp
            )
            sys.exit(0)
            
        elif args.command == "verify":
            ok = verify_sorted(args.file)
            sys.exit(0 if ok else 1)
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"未知错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
