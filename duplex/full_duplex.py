import re
import asyncio

from .align_asr import (
    build_buffer_and_prompt,
    construct_user_prompt,
    convert_to_label,
    get_segments,
    merge_chunks,
    prepare_intent,
)
from .service_config import (
    BACKCHANNEL_WORDS,
    MAX_BACK,
    MIN_INFER_NUM_WORD,
    PARTICLES,
    SPECIAL_TEXT,
    assistant_prompt,
    system_prompt,
    trunc_delimiter,
    user_prompt,
)


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def fallback_handler(asr_text, vad_final):
    if vad_final:
        output_text = "<valid><|im_end|>"
    elif len(asr_text) >= MIN_INFER_NUM_WORD:
        output_text = "<valid>"
    else:
        output_text = "fallback"

    return output_text

def can_match(s, word_dict):
    """
    判断字符串 s 是否能由词典中的词覆盖（完整匹配或前缀匹配）。

    规则：
    - 所有词完整匹配 -> True
    - 前面词完整匹配 + 最后一个词是词典中某词的前缀 -> True
    - 否则 -> False
    """
    if not s:
        return True

    n = len(s)

    # dp[i] = s[0:i] 能否被完整拆分
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for word in word_dict:
            wlen = len(word)
            if i >= wlen and dp[i - wlen] and s[i - wlen:i] == word:
                dp[i] = True
                break

    # 完整匹配
    if dp[n]:
        return True

    # 前缀匹配：完整匹配前缀 + 剩余部分是某词典词的前缀
    for i in range(n + 1):
        if dp[i] and i < n:
            remaining = s[i:]
            for word in word_dict:
                if word.startswith(remaining):
                    return True

    return False

async def turn_end_logging(data):
    round_id = data.roundId
    history = data.history
    history = _model_to_dict(history) if history is not None else None

    vad_final = data.input.vadFinal
    dual_vad = True if data.input.dualVad else False

    ret = {"history": history}

    if history is not None:
        last_round_id = round_id
        prev_round_id = round_id
        prompt_locked = history["context"]
        buffer = history["buffer"]
        temp_buffer = history["tempBuffer"]
        text_to_concat = history["textToConcat"]
        text_to_trim = history["textToTrim"]
        clean_past = history["cleanPast"]
        last_output = history["lastOutput"]
        is_semantic_complete = history["isSemComplete"]
        timer_type = history["timerType"]
        reset_timer = False

        log_data = history["logData"]

        user_answer = len(buffer) + len(temp_buffer) > 0
        timer_type = 0
        text_to_concat = ""
        is_semantic_complete = False
        last_output = "" # AI刚做出回应后重置，避免重复响应短超时信号

        if user_answer:
            total_buffer = temp_buffer + buffer
            buffer_text = "".join([chunk["user"] + chunk["belongTo"] for chunk in total_buffer])
            if log_data:
                log_data['bufferText'] = buffer_text
            prompt_editable = construct_user_prompt(total_buffer)
            if prompt_locked.endswith(assistant_prompt):
                prompt_locked = prompt_locked.removesuffix(assistant_prompt)
                prompt_editable = prompt_editable.removeprefix(user_prompt[:-2])
            prompt = prompt_locked + prompt_editable
            prompt_locked = prompt

        log_output = log_data
        if log_output is not None:
            replies = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", prompt_locked, re.S)
            log_output["lastReply"] = replies[-1] if len(replies) > 0 else ""
            log_output["labelText"] = convert_to_label(log_data['bufferText'], log_data['endType'])

        if user_answer and not prompt_locked.endswith(assistant_prompt):
            prompt_locked += assistant_prompt

        state = 0
        final_text = []
        if len(buffer) > 0:
            state = 2  # end强行返回state=2
            final_text = prepare_intent(buffer) # 当前对话转成修正后的字符串
            if not dual_vad:
                text_to_trim.append(final_text[-1] if len(final_text) > 0 else "")

        buffer = []
        temp_buffer = []
        log_data = None

        ret = {
            "state": state,
            "finalText": final_text,
            "history": {
                "lastRoundId": last_round_id,
                "prevRoundId": prev_round_id,
                "context": prompt_locked,
                "buffer": buffer,
                "tempBuffer": temp_buffer,
                "textToConcat": text_to_concat,
                "textToTrim": text_to_trim,
                "cleanPast": clean_past,
                "lastOutput": last_output,
                "isSemComplete": is_semantic_complete,
                "timerType": timer_type,
                "resetTimer": reset_timer,
                "logData": log_data,
            },
            'logInfo': log_output
        }

    return ret

async def get_duplex_response(triton_client, data, max_context_len, timeout):
    fallback = False
    fallback_msg = ""

    if not await triton_client.health_check(): # triton服务不健康，回退兜底
        fallback = True
        fallback_msg = "LMDeploy gRPC server is not healthy"

    state = 0
    interrupt = False
    output = ""
    log_output = None

    round_id = data.roundId
    dual_vad = True if data.input.dualVad else False

    tts_text = data.input.ttsText
    asr_text = data.input.asrText
    start_time = data.input.startTime
    end_time = data.input.endTime
    vad_final = data.input.vadFinal
    time_out = data.input.timeOut

    history = data.history

    # 初始化history
    if history is None:
        history = {
            "lastRoundId": None,
            "prevRoundId": None,
            "context": "",
            "buffer": [],
            "tempBuffer": [],
            "textToConcat": "",
            "textToTrim": [],
            "cleanPast": False,
            "lastOutput": "",
            "isSemComplete": False,
            "timerType": 0,
            "resetTimer": False,
            "logData": None,
        }
    else:
        history = _model_to_dict(history)

    last_round_id = history['lastRoundId']
    prev_round_id = history['prevRoundId']
    prompt_locked = history["context"]
    buffer = history["buffer"]
    temp_buffer = history["tempBuffer"]
    text_to_concat = history["textToConcat"]
    text_to_trim = history["textToTrim"]
    clean_past = history["cleanPast"]
    last_output = history["lastOutput"]
    is_semantic_complete = history["isSemComplete"]
    timer_type = history["timerType"]
    last_timer_type = timer_type
    reset_timer = False
    is_special = asr_text == SPECIAL_TEXT

    log_data = history["logData"]
    vad_batch = log_data['vadBatch'] if log_data else []
    time_batch = log_data['timeBatch'] if log_data else []
    count_batch = log_data['countBatch'] if log_data else []

    if len(data.blackList) == 0:
        data.blackList = BACKCHANNEL_WORDS
    interrupt_pattern = re.compile(rf"(?:{'|'.join(map(re.escape, data.whiteList)) or r'(?!)'})+")
    particle_pattern = re.compile(rf"^({'|'.join(map(re.escape, PARTICLES))})+$")

    if clean_past:
        text_to_trim = []
        clean_past = False

    if prev_round_id is not None and prev_round_id != round_id:  # 新轮次
        user_answer = len(buffer) + len(temp_buffer) > 0
        last_round_id = prev_round_id
        timer_type = 0
        text_to_concat = ""
        is_semantic_complete = False
        last_output = "" # AI刚做出回应后重置，避免重复响应短超时信号

        if user_answer: # 被强制结束轮次（上一轮语音流已经结束了，传进的内容一定是下一轮的）
            total_buffer = temp_buffer + buffer
            buffer_text = "".join([chunk["user"] + chunk["belongTo"] for chunk in total_buffer])
            if log_data:
                log_data['bufferText'] = buffer_text
            prompt_editable = construct_user_prompt(total_buffer)
            if prompt_locked.endswith(assistant_prompt):
                prompt_locked = prompt_locked.removesuffix(assistant_prompt)
                prompt_editable = prompt_editable.removeprefix(user_prompt[:-2])
            prompt = prompt_locked + prompt_editable
            prompt_locked = prompt

        log_output = log_data
        if log_output is not None:
            replies = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", prompt_locked, re.S)
            log_output["lastReply"] = replies[-1] if len(replies) > 0 else ""
            log_output["labelText"] = convert_to_label(log_data['bufferText'], log_data['endType'])

        buffer = []
        temp_buffer = []
        vad_batch = []
        time_batch = []
        count_batch = []
        log_data = None

        if user_answer and not prompt_locked.endswith(assistant_prompt):
            prompt_locked += assistant_prompt

    prev_round_id = round_id

    if asr_text is not None:   # 正常asr中间结果
        # 处理AI内容
        tts_text = "" if tts_text is None else tts_text

        if prompt_locked == "":
            prompt_locked += system_prompt
            if tts_text != "": # 有开头语时
                prompt_locked += assistant_prompt

        prompt_locked += tts_text # 播报的一定是上一轮的内容

        # 处理用户内容
        if len(text_to_concat) > 0:
            asr_text = text_to_concat + asr_text

        if len(text_to_trim) > 0:
            if is_special:
                text_to_trim = ["" for _ in range(len(text_to_trim))]
            else:
                text_to_trim, asr_text = get_segments(text_to_trim, asr_text)

        # buffer: 当前轮次; prompt_editable: 历史上下文; asr_text: 新增输入
        if is_special:
            asr_text = asr_text.replace(SPECIAL_TEXT, "")

        buffer, asr_text, back_step = build_buffer_and_prompt(buffer, asr_text, max_back=MAX_BACK)
        update_old_info = False
        # 存在历史识别变动，进行回溯推理
        if back_step > 0:
            asr_text = "".join([b['user'] for b in buffer][-back_step:]) + asr_text
            t_buffer = buffer[:-back_step]

            if asr_text == "" and back_step < len(buffer):
                back_step += 1
                update_old_info = len(buffer) > 0

            buffer = t_buffer
            vad_batch = vad_batch[:-back_step]
            time_batch = time_batch[:-back_step]
            count_batch = count_batch[:-back_step]

        if len(text_to_trim) > 0:
            past_asr_text = "".join(text_to_trim)
            temp_idx = 0
            for b in reversed(temp_buffer):
                if b['isBoundary']:
                    break
                temp_idx += 1
            _temp_buffer = temp_buffer[-temp_idx:]
            _temp_buffer, _, _ = build_buffer_and_prompt(_temp_buffer, past_asr_text)
            temp_buffer = temp_buffer[:-temp_idx] + _temp_buffer

        prompt_editable = construct_user_prompt(temp_buffer + buffer)

        prompt_locked_ = prompt_locked
        prompt_editable_ = prompt_editable
        if prompt_locked.endswith(assistant_prompt) and tts_text == "": # 轮次结束但没有tts播报文本
            prompt_locked_ = prompt_locked.removesuffix(assistant_prompt)
            prompt_editable_ = prompt_editable.removeprefix(user_prompt[:-2])
        prompt = prompt_locked_ + prompt_editable_

        # context过长处理，去除前面轮次的交互
        expected_len = len(prompt)
        if asr_text != "":
            expected_len += len(asr_text)

        if expected_len >= max_context_len - 1:
            diff = expected_len - (max_context_len - 1)
            matches = [m.start() for m in re.finditer(re.escape(trunc_delimiter), prompt_locked)]
            if len(matches) > 0:
                turn_i = matches[0]
                accu_l = 0
                for m in matches[1:]:
                    accu_l += m - turn_i
                    turn_i = m
                    if accu_l > diff:
                        break

                prompt_locked = prompt_locked[:matches[0]] + prompt_locked[turn_i:]
                prompt = prompt[:matches[0]] + prompt[turn_i:]

                if accu_l <= diff: # 从prompt_locked中截断，二次确认长度
                    return {"error": "Context too long"}, None
            else:
                return {"error": "Context too long"}, None

        # asr结果更新
        if asr_text != "":
            cur_asr_text = asr_text
            for b in reversed(buffer):
                if b['isBoundary']:
                    break
                cur_asr_text = b['user'] + cur_asr_text
            if not vad_final and len(cur_asr_text) < MIN_INFER_NUM_WORD:   # 句子的开头若干字不进行有效性判断
                state = 0

                buffer.append({
                    "user": asr_text,
                    "predict": "<skip>",
                    "belongTo": "<uncertain>",
                    "isBoundary": False
                })
                last_output = "<uncertain>"

                reset_timer = False
                timer_type = 4

            else:
                buffer.append({
                    "user": asr_text,
                    "predict": "",
                    "belongTo": "",
                    "isBoundary": False
                })
                prompt += asr_text
                # print(prompt.replace('\n', '\\n'))

                # 95511猜你想问功能临时兜底处理
                ai_text = re.findall(r"<\|im_start\|>assistant\n(.*?)$", prompt_locked, re.S)
                ai_text = ai_text[-1] if len(ai_text) > 0 else ""
                guess_pattern = re.compile(r"(你好，中国平安|您好，中国平安).*?查询到您(?!.*哪个)")
                guess_flag = guess_pattern.search(ai_text)

                if fallback:
                    output_text = fallback_handler(data.input.asrText, vad_final)
                elif (
                    (guess_flag or '是否' in ai_text)
                    and (
                        '是' in cur_asr_text
                        or '否' in cur_asr_text
                        or '要' in cur_asr_text
                        or ('继续' in ai_text and '继续' in cur_asr_text)
                    )
                ):
                    output_text = "<valid><|im_end|>"
                elif interrupt_pattern.search(cur_asr_text):
                    interrupt = True
                    fallback = True
                    fallback_msg = "Hit words in the whitelist"
                    output_text = "<valid><|im_end|>"
                elif particle_pattern.search(cur_asr_text):
                    fallback = True
                    fallback_msg = "Pure particles"
                    output_text = "<invalid>"
                else:
                    try:
                        output_text = ""
                        validity_output = await triton_client.infer(
                            data.requestId,
                            prompt,
                            decoding_type=0,
                            timeout=timeout,
                        )
                        if validity_output != "<valid>" and validity_output != "<invalid>":
                            validity_output = ""

                        output_text += validity_output
                        prompt_t = prompt + validity_output

                        # 确认有效后判断语义完整性
                        if validity_output == "<valid>":
                            completion_output = await triton_client.infer(
                                data.requestId,
                                prompt_t,
                                decoding_type=1,
                                timeout=timeout,
                            )
                            output_text += completion_output

                    except Exception as e:
                        fallback = True
                        if type(e) == asyncio.TimeoutError:
                            e = "timeout"
                        fallback_msg = f"LMDeploy gRPC request failed with: {e}"
                        output_text = fallback_handler(data.input.asrText, vad_final)

                output = ""
                if output_text.startswith('<invalid>'):
                    output = "<invalid>"
                elif output_text.startswith('<valid>'):
                    output = "<valid>"
                    is_semantic_complete = output_text == "<valid><|im_end|>"
                prompt += output

                buffer[-1]["predict"] = output
                buffer[-1]["belongTo"] = output if output != "" else "<uncertain>"

                backchannel = (not interrupt) and can_match(
                    "".join(b['user'] for b in buffer if b['belongTo'] != "<invalid>"),
                    data.blackList,
                )

                if output == "<valid>":
                    state = 1 if not backchannel else 0
                    if last_output == "<uncertain>":
                        for b in reversed(buffer[:-1]):
                            if b["belongTo"] == "<uncertain>":
                                b["belongTo"] = "<valid>"
                            else:
                                break
                elif output == "<invalid>":
                    state = 0
                    if last_output == "<uncertain>":
                        for b in reversed(buffer[:-1]):
                            if b["belongTo"] == "<uncertain>":
                                b["belongTo"] = "<invalid>"
                            else:
                                break

                if is_semantic_complete:   # 语义完整，AI可回复
                    if output == "<valid>":
                        reset_timer = True
                        if not vad_final:
                            timer_type = 1 if not backchannel else 6  # 语义完整后持续无效（包含有字/无字）兜底
                        else:
                            timer_type = 2 if not backchannel else 7 # 等待一个气口减去vad delay时间
                    else:
                        reset_timer = False
                        if not vad_final:   # 无效输入，继续计时，不重置
                            pass
                        else:   # 持续无效后遇到vad final
                            timer_type = 2 if not backchannel else 7
                else:   # 语义不完整
                    if vad_final:   # 语音结束开始计时
                        reset_timer = True
                        if output == "<valid>": # 语义不完整的有效问询，相比语义完整的多等一会儿
                            timer_type = 3 if not backchannel else 7 # 等待一个思考时间
                        else:   # 无效音频片段结束，多等待一会儿用户输入
                            timer_type = 4
                        if last_timer_type == 4:
                            reset_timer = False
                    else:   # 语音未结束
                        if output == "<valid>":
                            timer_type = 0 # 有效输出，不计时（包含超长超时）
                        else:
                            timer_type = 4  # type=4 & reset=false表明需要进行超长超时， 当type!=4时结束计时

                last_output = buffer[-1]["belongTo"]

        else:   # 无新增识别结果（vad final或历史识别纠正）
            if fallback:
                is_semantic_complete = vad_final

            if len(buffer) > 0:
                backchannel = can_match(
                    "".join(b['user'] for b in buffer if b['belongTo'] != "<invalid>"),
                    data.blackList,
                )

                if is_semantic_complete:   # 语义完整，AI可回复
                    if vad_final:   # 语义完整后获取到vad final信号，只等待最多timer=1的时间
                        reset_timer = False
                        state = 1 if not backchannel else 0
                        timer_type = 2 if not backchannel else 7
                    else:
                        pass
                else:   # 语义不完整
                    if vad_final:  # 语音结束开始计时
                        reset_timer = True
                        if last_output == "<valid>": # 语义不完整的有效问询，相比语义完整的多等一会儿
                            state = 1 if not backchannel else 0
                            timer_type = 3 if not backchannel else 7 # 等待一个思考时间
                        elif last_output in ("<invalid>", "<uncertain>", ""):   # 无效音频片段结束，多等待一会儿用户输入
                            timer_type = 4
                    else:   # 语音未结束
                        pass
            else:   # 纠错后是空内容
                state = 0
                timer_type = 4
                if vad_final and last_timer_type != 4:
                    reset_timer = True

        if is_special:
            buffer.append({
                "user": SPECIAL_TEXT,
                "predict": "<invalid>",
                "belongTo": "<invalid>",
                "isBoundary": True,
            })
            last_output = "<invalid>"

        if vad_final:
            if not dual_vad:
                text_to_concat = "".join([b['user'] for b in buffer])  # 暂存之前vad语义未结束的text
                clean_past = True   # 语义轮次结束，下一轮不会遇到文本裁剪
            if len(buffer) > 0:
                buffer[-1]['isBoundary'] = True

        if asr_text != "" or update_old_info:
            vad_batch.append(vad_final)
            time_batch.append((start_time, end_time))
            count_batch.append(len(temp_buffer) + len(buffer))

            end_type = 0

        else:  # timeout
            if time_out > 0:
                state = 2

            timer_type = 0
            text_to_concat = ""

            is_semantic_complete = False
            last_output = "" # AI刚做出回应后重置，避免重复响应短超时信号

            if time_out in (1, 2, 6, 7):
                end_type = 1
            elif time_out == 3:
                end_type = 2
            else:
                end_type = 0

    temp_buffer, buffer, vad_batch, time_batch, count_batch = merge_chunks(
        temp_buffer,
        buffer,
        vad_batch,
        time_batch,
        count_batch,
    )
    final_text = prepare_intent(buffer) # 当前对话转成修正后的字符串

    if len(vad_batch) == 0:
        log_data = None
    else:
        log_data = {
            "vadBatch": vad_batch,
            "timeBatch": time_batch,
            "countBatch": count_batch,
            "endType": end_type,
        }

    if state == 2:
        if not dual_vad:
            temp_buffer.extend(buffer)
            buffer = []

            if not vad_final:
                text_to_trim.append(final_text[-1] if len(final_text) > 0 else "")

    ret = {
        "state": state,
        "finalText": final_text,
        "history": {
            "lastRoundId": last_round_id,
            "prevRoundId": prev_round_id,
            "context": prompt_locked,
            "buffer": buffer,
            "tempBuffer": temp_buffer,
            "textToConcat": text_to_concat,
            "textToTrim": text_to_trim,
            "cleanPast": clean_past,
            "lastOutput": last_output,
            "isSemComplete": is_semantic_complete,
            "timerType": timer_type,
            "resetTimer": reset_timer,
            "logData": log_data,
        }
    }

    if log_output:
        ret.update({'logInfo': log_output})

    if fallback:
        ret.update({'fallbackMsg': fallback_msg})

    return ret
