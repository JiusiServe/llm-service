import os
from vllm.distributed.kv_events import KVEventBatch as KVEventBatchOri


class KVEventBatch(KVEventBatchOri):

    vllm_instance_id: str = ""

    def __post_init__(self):
        self.vllm_instance_id = os.getenv("VLLM_INSTANCE_ID", "")


def monkey_patch():
    vllm.distributed.kv_events.KVEventBatchOri = vllm.distributed.kv_events.KVEventBatch
    vllm.distributed.kv_events.KVEventBatch = KVEventBatch
