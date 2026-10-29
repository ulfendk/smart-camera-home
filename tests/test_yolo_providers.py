from smartcam.detect.yolo import providers


def test_openvino_first_with_its_device():
    names, options = providers(["CPUExecutionProvider", "OpenVINOExecutionProvider"], "NPU")
    assert names == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options == [{"device_type": "NPU"}, {}]


def test_cuda_image_unchanged():
    names, options = providers(["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"])
    assert names == ["CUDAExecutionProvider", "CPUExecutionProvider"] and options == [{}, {}]
