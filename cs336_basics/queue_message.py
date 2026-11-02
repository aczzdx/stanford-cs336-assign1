from typing import Any, TypedDict


class QueueMsg(TypedDict):
    action_type: str
    payload: Any
    
if __name__ == '__main__':
    import pickle as pkl
    msg = {"action_type": "done", "payload": None}
    msg2: QueueMsg = {"action_type": "done", "payload": None}

    def _size(x):
        return len(pkl.dumps(x))
    
    print(f"msg size = {_size(msg)}")
    print(f"msg2 size = {_size(msg2)}")

    for i in range(100):
        print(i)
        match i:
            case 3:
                break
    
