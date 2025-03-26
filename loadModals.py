# 下载模型 (选择一种)
# tiny.en - 最小的英语模型（约39M）
# base.en - 基础英语模型（约74M）
# small.en - 小型英语模型（约244M）
# medium.en - 中型英语模型（约769M）
# large - 最大的多语言模型（约2.9G）

import whisper

models = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
]

# 例如，下载medium模型
model = whisper.load_model("large-v3-turbo")

# 如果需要中文支持，建议使用不带.en后缀的模型
# model = whisper.load_model('medium')
