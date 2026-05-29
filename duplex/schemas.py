from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .service_config import BACKCHANNEL_WORDS


class BufferData(BaseModel):
    user: str
    predict: Literal["", "<valid>", "<invalid>", "<skip>"]
    belongTo: Literal["<valid>", "<invalid>", "<uncertain>"]
    isBoundary: bool


class LogData(BaseModel):
    vadBatch: list[bool]
    timeBatch: list[tuple[float, float]]
    countBatch: list[int]
    endType: Literal[0, 1, 2]


class HistoryData(BaseModel):
    lastRoundId: str | None = None
    prevRoundId: str | None = None
    context: str
    buffer: list[BufferData]
    tempBuffer: list[BufferData]
    textToConcat: str
    textToTrim: list[str]
    cleanPast: bool
    lastOutput: Literal["", "<valid>", "<invalid>", "<uncertain>"]
    isSemComplete: bool
    timerType: Literal[0, 1, 2, 3, 4, 6, 7]
    resetTimer: bool
    logData: LogData | None = None


class InputData(BaseModel):
    ttsText: str | None = None
    asrText: str | None = None
    startTime: float | None = None
    endTime: float | None = None
    vadFinal: bool | None = None
    dualVad: bool | None = None
    timeOut: Literal[0, 1, 2, 3, 4, 5, 6, 7] | None = None


class EndData(BaseModel):
    callId: str
    sessionId: str
    requestId: str
    roundId: str
    input: InputData
    history: HistoryData | None = None


class InferData(BaseModel):
    callId: str
    sessionId: str
    requestId: str
    roundId: str
    input: InputData
    history: HistoryData | None = None
    whiteList: list[str] = Field(default_factory=list)
    blackList: list[str] = Field(default_factory=lambda: list(BACKCHANNEL_WORDS))


def model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
