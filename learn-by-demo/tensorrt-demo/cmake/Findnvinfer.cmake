find_library(nvinfer_LIBRARY nvinfer REQUIRED)
find_path(nvinfer_INCLUDE_DIR NvInfer.h REQUIRED)
add_library(nvinfer::nvinfer UNKNOWN IMPORTED)
set_target_properties(nvinfer::nvinfer PROPERTIES
    IMPORTED_LOCATION "${nvinfer_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${nvinfer_INCLUDE_DIR}"
)

find_library(onnxparser_LIBRARY nvonnxparser REQUIRED)
find_path(onnxparser_INCLUDE_DIR NvOnnxParser.h REQUIRED)
add_library(nvinfer::onnxparser UNKNOWN IMPORTED)
set_target_properties(nvinfer::onnxparser PROPERTIES
    IMPORTED_LOCATION "${onnxparser_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxparser_INCLUDE_DIR}"
)

