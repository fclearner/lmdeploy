import difflib
import re

from .service_config import user_prompt, INVALID_START, INVALID_END, ALIGN_TOKEN, SPECIAL_TEXT


def adjust_replace_opcodes(ops, old_text, new_text):
    new_ops = []
    for tag, i1, i2, j1, j2 in ops:
        # 若旧文本替换部分比新文本长，则拆分为单字符替换和后续删除
        if tag == "replace" and (i2 - i1) > (j2 - j1):
            d = (i2 - i1) - (j2 - j1)
            new_ops.append(('delete', i1, i1+d, j1, j1))
            new_ops.append(('replace', i1+d, i2, j1, j2))
        else:
            new_ops.append((tag, i1, i2, j1, j2))
    return new_ops


def build_alignment(old_text, new_text):
    matcher = difflib.SequenceMatcher(None, old_text, new_text)
    ops = adjust_replace_opcodes(matcher.get_opcodes(), old_text, new_text)
    a_old, a_new = [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("equal", "replace"):
            for k in range(i2 - i1):
                a_old.append(old_text[i1 + k])
                if k == i2 - i1 - 1 and j2 >= i2:
                    a_new.append(new_text[j1 + k])
                    if j2 - j1 - k > 1:
                        a_old.append(ALIGN_TOKEN)
                        a_new.append(new_text[j1 + k + 1:j2])
                else:
                    a_new.append(new_text[j1 + k])
        elif tag == "delete":
            a_old.extend(old_text[i1:i2])
            a_new.extend(ALIGN_TOKEN * (i2 - i1))
        elif tag == "insert":
            a_old.extend(ALIGN_TOKEN * (j2 - j1))
            a_new.extend(new_text[j1:j2])
    return a_old, a_new


def build_mapping(a_old):
    mapping = []
    for i, o in enumerate(a_old):
        if o != ALIGN_TOKEN:
            mapping.append(i)

    return mapping


def remove_align_tokens(s):
    return [x for x in s if x != ALIGN_TOKEN]


def extract_positions(texts):
    end_pos = []
    invalid_pos = []
    is_empty = []
    pos = 0

    for seg in texts:
        # 处理当前段的每个字符
        start = 0

        for char in seg:
            if char == INVALID_START:
                start = pos
            elif char == INVALID_END:
                invalid_pos.append((start, pos - 1))
            else:
                pos += 1

        # 记录分组位置
        end_pos.append(pos - 1)
        is_empty.append(seg == "")

    return end_pos, invalid_pos, is_empty


def postprocess_str(s):
    s = s.replace(ALIGN_TOKEN, "")
    return re.sub(
        re.escape(INVALID_END)
        + r'('
        + re.escape(INVALID_START)
        + r'|'
        + re.escape(INVALID_END)
        + r')*'
        + re.escape(INVALID_START),
        "",
        re.sub(re.escape(INVALID_START + INVALID_END), "", s)
    )


def generate_segments(texts, ends, invalids=[]):
    for start, end in invalids:
        texts[start] = INVALID_START + texts[start]
        texts[end] = texts[end] + INVALID_END

    ends = [-1] + ends

    if len(texts) - 1 in ends:
        return [postprocess_str("".join(texts[ends[i]+1: ends[i+1]+1])) for i in range(len(ends)-1)], ""
    else:
        ends = ends + [len(texts) - 1]
        all_segs = [postprocess_str("".join(texts[ends[i]+1: ends[i+1]+1])) for i in range(len(ends)-1)]
        return all_segs[:-1], all_segs[-1]


def get_segments(trim_segments, raw_text):
    end_pos, invalid_pos, is_empty = extract_positions(trim_segments)

    trim_str = "".join(trim_segments).replace(INVALID_START, '').replace(INVALID_END, '')

    a_trim, a_raw = build_alignment(trim_str, raw_text)

    mapping = build_mapping(a_trim)

    new_end_pos, new_invalid_pos = [], []
    if len(mapping) > 0:
        new_end_pos = [mapping[pos] for i, pos in enumerate(end_pos) if not is_empty[i]]
        new_invalid_pos = [(mapping[s], mapping[e]) for (s, e) in invalid_pos]

    past_segs, cur_seg = generate_segments(a_raw, new_end_pos, new_invalid_pos)

    it = iter(past_segs)
    past_segs = ["" if is_empty[i] else next(it, "") for i in range(len(trim_segments))]

    return past_segs, cur_seg


def build_buffer_and_prompt(old_chunks, new_asr, max_back=-1):
    # 拼接所有用户文本构成旧用户纯文本，同时记录每轮累计长度边界
    old_asr = "".join(chunk["user"] for chunk in old_chunks)
    end_pos, _, is_empty = extract_positions([chunk["user"] for chunk in old_chunks])

    new_asr = new_asr.replace(INVALID_START, '').replace(INVALID_END, '')

    # 对齐旧用户文本与新ASR文本，构造位置映射
    a_old, a_new = build_alignment(old_asr, new_asr)
    mapping = build_mapping(a_old)
    new_end_pos = []
    if len(mapping) > 0:
        new_end_pos = [mapping[pos] for i, pos in enumerate(end_pos) if not is_empty[i]]

    # 将新ASR文本切分为各轮
    past_chunks, cur_chunk = generate_segments(a_new, new_end_pos)

    it = iter(past_chunks)
    past_chunks = ["" if is_empty[i] else next(it, "") for i in range(len(old_chunks))]

    if cur_chunk != "":
        past_chunks.append(cur_chunk)

    # 生成纠正过的buffer和prompt
    buffer = []
    traceback = 0
    for chunk in reversed(old_chunks):
        if chunk['isBoundary'] or chunk['predict'] != "<skip>":
            break
        else:
            traceback += 1

    L = len(old_chunks)
    max_back = L if max_back == -1 else max_back
    index_found = False
    for idx, (o_chunk, new_user) in enumerate(zip(old_chunks, past_chunks)):
        if not index_found and o_chunk['user'] != new_user:
            traceback = max(traceback, min(max_back, L - idx))
            index_found = True
        buffer.append({
            "user": new_user,
            "predict": o_chunk["predict"],
            "belongTo": o_chunk["belongTo"],
            "isBoundary": o_chunk["isBoundary"]
        })

    return buffer, cur_chunk, traceback


def construct_user_prompt(buffer):
    prompt = []
    for chunk in buffer:
        if chunk["user"] == SPECIAL_TEXT or chunk["user"] == "":
            continue
        prompt.append(chunk["user"] + chunk["predict"])

    return user_prompt.format("".join(prompt))


def prepare_intent(buffer):
    intent = []

    left, right = INVALID_START, INVALID_END

    prev = None
    tmp = ""
    for i, b in enumerate(buffer):
        if b['user'] != "":
            cur = b["belongTo"]

            if cur != prev:
                if cur in ("<valid>", "<uncertain>"):  # 保守<uncertain>当做有效
                    if prev == "<invalid>":
                        tmp += right + b["user"]
                    else:
                        tmp += b["user"]
                else:
                    tmp += left + b["user"]
            else:
                tmp += b["user"]

            if (b['isBoundary'] or i == len(buffer) - 1) and cur in ("<invalid>"):
                tmp += right

            prev = cur

        if b['isBoundary'] or i == len(buffer) - 1:
            if tmp != "":
                intent.append(tmp)
            tmp = ""
            prev = None

    return intent


def merge_chunks(
    temp_buffer,
    buffer,
    vad_batch,
    time_batch,
    count_batch
):
    if not count_batch:
        return list(temp_buffer), list(buffer), [], [], []

    assert len(vad_batch) == len(time_batch) == len(count_batch), (
        f"批次长度不一致: vad={len(vad_batch)}, time={len(time_batch)}, count={len(count_batch)}"
    )

    n = len(vad_batch)
    t_len = len(temp_buffer)
    b_len = len(buffer)
    m = t_len + b_len

    # 验证count_batch的最后一个值是否与总长度匹配
    last_count = count_batch[-1] if count_batch else 0
    if last_count != m:
        raise ValueError(
            f"count_batch最后一个值({last_count}) != temp_buffer+buffer总长度({m}). "
            f"如果temp_buffer已被清理过，请将空chunk补回或修正count_batch."
        )

    # ===== Step 1: 统一清理空chunk, isBoundary向前合并 =====
    all_chunks_raw = [dict(c) for c in temp_buffer] + [dict(c) for c in buffer]

    keep_indices = []
    for i in range(m):
        chunk = all_chunks_raw[i]
        if chunk.get('user', '') == '':
            if chunk.get('isBoundary', False) and keep_indices:
                all_chunks_raw[keep_indices[-1]]['isBoundary'] = True
        else:
            keep_indices.append(i)

    if not keep_indices:
        return [], [], [], [], []

    keep_chunks = [all_chunks_raw[i] for i in keep_indices]
    keep_m = len(keep_chunks)

    # 记录切分点: 保留的chunk中, 有多少个属于temp_buffer范围
    new_t_len = sum(1 for idx in keep_indices if idx < t_len)

    # ===== Step 2: 预处理 first_appear =====
    first_appear = [-1] * m
    ri = 0
    for oi in range(m):
        while ri < n and count_batch[ri] < oi + 1:
            ri += 1
        first_appear[oi] = ri if ri < n else n - 1

    # next_vad[j]: 从位置 j 开始，第一个 vad=True 的索引
    next_vad = [n] * (n + 1)
    for j in range(n - 1, -1, -1):
        next_vad[j] = j if vad_batch[j] else next_vad[j + 1]

    # ===== Step 3: 为每个保留的chunk找确认请求 =====
    all_vad, all_time, all_count = [], [], []

    for ki in range(keep_m):
        orig_idx = keep_indices[ki]
        chunk = keep_chunks[ki]
        fa = first_appear[orig_idx]
        fa_count = count_batch[fa] if fa < n else count_batch[-1]

        if chunk.get('isBoundary', False):
            ci = next_vad[fa] if next_vad[fa] < n else fa
        else:
            end = next_vad[fa] if next_vad[fa] < n else n
            ci = fa
            for j in range(fa, end):
                if count_batch[j] == fa_count:
                    ci = j

        ci = max(0, min(ci, n - 1))
        vad = vad_batch[ci]

        if vad:
            chunk['isBoundary'] = True

        all_vad.append(vad)
        all_time.append(time_batch[ci])
        all_count.append(ki + 1)

    # 按 new_t_len 切分
    temp_out = keep_chunks[:new_t_len]
    buf_out = keep_chunks[new_t_len:]

    return temp_out, buf_out, all_vad, all_time, all_count


def convert_to_label(user_str, end_type):
    parts = re.findall(r"(.*?)(<valid>|<invalid>|<uncertain>)", user_str)
    parts = [(txt, tag) for txt, tag in parts if txt or tag]

    # 处理 uncertain
    resolved = []
    n = len(parts)
    for i, (text, tag) in enumerate(parts):
        if tag == "<uncertain>":
            # 查找未来第一个非 uncertain
            j = i + 1
            next_tag = None
            while j < n:
                if parts[j][1] != "<uncertain>":
                    next_tag = parts[j][1]
                    break
                j += 1
            resolved_tag = next_tag if next_tag is not None else "<invalid>"
        else:
            resolved_tag = tag
        resolved.append((text, resolved_tag))

    # 合并成 <有效> / <无效>
    blocks = []
    for text, tag in resolved:
        simple_tag = "valid" if tag == "<valid>" else "invalid"

        if not blocks or blocks[-1]["tag"] != simple_tag:
            blocks.append({"tag": simple_tag, "text": text})
        else:
            blocks[-1]["text"] += text

    # 处理最后一个有效块
    suffix = ""
    if end_type == 1:
        suffix = "&lt;结束&gt;"
    elif end_type == 2:
        suffix = "&lt;未结束&gt;"
    for b in reversed(blocks):
        if b["tag"] == "valid":
            b["text"] += suffix
            break

    return "".join(
        f"&lt;有效&gt;{b['text']}&lt;/有效&gt;" if b["tag"] == "valid"
        else f"&lt;无效&gt;{b['text']}&lt;/无效&gt;"
        for b in blocks
    )
