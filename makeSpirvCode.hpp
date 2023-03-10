#include <array>

constexpr auto makeSpirvCode(const u_int32_t bufferLength)
{
    enum : u_int32_t
    {
        RESERVED_ID = 0,
        FUNC_ID,
        IN_ID,
        OUT_ID,
        GLOBAL_INVOCATION_ID,
        VOID_TYPE_ID,
        FUNC_TYPE_ID,
        INT_TYPE_ID,
        INT_ARRAY_TYPE_ID,
        STRUCT_ID,
        POINTER_TYPE_ID,
        ELEMENT_POINTER_TYPE_ID,
        INT_VECTOR_TYPE_ID,
        INT_VECTOR_POINTER_TYPE_ID,
        INT_POINTER_TYPE_ID,
        CONSTANT_ZERO_ID,
        CONSTANT_ARRAY_LENGTH_ID,
        LABEL_ID,
        IN_ELEMENT_ID,
        OUT_ELEMENT_ID,
        GLOBAL_INVOCATION_X_ID,
        GLOBAL_INVOCATION_X_PTR_ID,
        TEMP_LOADED_ID,
        BOUND
    };

    enum : u_int32_t
    {
        INPUT = 1,
        UNIFORM = 2,
        BUFFER_BLOCK = 3,
        ARRAY_STRIDE = 6,
        BUILTIN = 11,
        BINDING = 33,
        OFFSET = 35,
        DESCRIPTOR_SET = 34,
        GLOBAL_INVOCATION = 28,
        OP_TYPE_VOID = 19,
        OP_TYPE_FUNCTION = 33,
        OP_TYPE_INT = 21,
        OP_TYPE_VECTOR = 23,
        OP_TYPE_ARRAY = 28,
        OP_TYPE_STRUCT = 30,
        OP_TYPE_POINTER = 32,
        OP_VARIABLE = 59,
        OP_DECORATE = 71,
        OP_MEMBER_DECORATE = 72,
        OP_FUNCTION = 54,
        OP_LABEL = 248,
        OP_ACCESS_CHAIN = 65,
        OP_CONSTANT = 43,
        OP_LOAD = 61,
        OP_STORE = 62,
        OP_RETURN = 253,
        OP_FUNCTION_END = 56,
        OP_CAPABILITY = 17,
        OP_MEMORY_MODEL = 14,
        OP_ENTRY_POINT = 15,
        OP_EXECUTION_MODE = 16,
        OP_COMPOSITE_EXTRACT = 81,
    };

    const u_int32_t shader[] = {
        // first is the SPIR-V header
        0x07230203, // magic header ID
        0x00010000, // version 1.0.0
        0,          // generator (optional)
        BOUND,      // bound
        0,          // schema

        // OpCapability Shader
        (2 << 16) | OP_CAPABILITY,
        1,

        // OpMemoryModel Logical Simple
        (3 << 16) | OP_MEMORY_MODEL,
        0,
        0,

        // OpEntryPoint GLCompute %FUNC_ID "f" %IN_ID %OUT_ID
        (5 << 16) | OP_ENTRY_POINT,
        5,
        FUNC_ID,
        0x66,
        GLOBAL_INVOCATION_ID,

        // OpExecutionMode %FUNC_ID LocalSize 1 1 1
        (6 << 16) | OP_EXECUTION_MODE,
        FUNC_ID,
        17,
        1,
        1,
        1,

        // next declare decorations

        (3 << 16) | OP_DECORATE,
        STRUCT_ID,
        BUFFER_BLOCK,

        (4 << 16) | OP_DECORATE,
        GLOBAL_INVOCATION_ID,
        BUILTIN,
        GLOBAL_INVOCATION,

        (4 << 16) | OP_DECORATE,
        IN_ID,
        DESCRIPTOR_SET,
        0,

        (4 << 16) | OP_DECORATE,
        IN_ID,
        BINDING,
        0,

        (4 << 16) | OP_DECORATE,
        OUT_ID,
        DESCRIPTOR_SET,
        0,

        (4 << 16) | OP_DECORATE,
        OUT_ID,
        BINDING,
        1,

        (4 << 16) | OP_DECORATE,
        INT_ARRAY_TYPE_ID,
        ARRAY_STRIDE,
        4,

        (5 << 16) | OP_MEMBER_DECORATE,
        STRUCT_ID,
        0,
        OFFSET,
        0,

        // next declare types
        (2 << 16) | OP_TYPE_VOID,
        VOID_TYPE_ID,

        (3 << 16) | OP_TYPE_FUNCTION,
        FUNC_TYPE_ID,
        VOID_TYPE_ID,

        (4 << 16) | OP_TYPE_INT,
        INT_TYPE_ID,
        32,
        1,

        (4 << 16) | OP_CONSTANT,
        INT_TYPE_ID,
        CONSTANT_ARRAY_LENGTH_ID,
        bufferLength,

        (4 << 16) | OP_TYPE_ARRAY,
        INT_ARRAY_TYPE_ID,
        INT_TYPE_ID,
        CONSTANT_ARRAY_LENGTH_ID,

        (3 << 16) | OP_TYPE_STRUCT,
        STRUCT_ID,
        INT_ARRAY_TYPE_ID,

        (4 << 16) | OP_TYPE_POINTER,
        POINTER_TYPE_ID,
        UNIFORM,
        STRUCT_ID,

        (4 << 16) | OP_TYPE_POINTER,
        ELEMENT_POINTER_TYPE_ID,
        UNIFORM,
        INT_TYPE_ID,

        (4 << 16) | OP_TYPE_VECTOR,
        INT_VECTOR_TYPE_ID,
        INT_TYPE_ID,
        3,

        (4 << 16) | OP_TYPE_POINTER,
        INT_VECTOR_POINTER_TYPE_ID,
        INPUT,
        INT_VECTOR_TYPE_ID,

        (4 << 16) | OP_TYPE_POINTER,
        INT_POINTER_TYPE_ID,
        INPUT,
        INT_TYPE_ID,

        // then declare constants
        (4 << 16) | OP_CONSTANT,
        INT_TYPE_ID,
        CONSTANT_ZERO_ID,
        0,

        // then declare variables
        (4 << 16) | OP_VARIABLE,
        POINTER_TYPE_ID,
        IN_ID,
        UNIFORM,

        (4 << 16) | OP_VARIABLE,
        POINTER_TYPE_ID,
        OUT_ID,
        UNIFORM,

        (4 << 16) | OP_VARIABLE,
        INT_VECTOR_POINTER_TYPE_ID,
        GLOBAL_INVOCATION_ID,
        INPUT,

        // then declare function
        (5 << 16) | OP_FUNCTION,
        VOID_TYPE_ID,
        FUNC_ID,
        0,
        FUNC_TYPE_ID,

        (2 << 16) | OP_LABEL,
        LABEL_ID,

        (5 << 16) | OP_ACCESS_CHAIN,
        INT_POINTER_TYPE_ID,
        GLOBAL_INVOCATION_X_PTR_ID,
        GLOBAL_INVOCATION_ID,
        CONSTANT_ZERO_ID,

        (4 << 16) | OP_LOAD,
        INT_TYPE_ID,
        GLOBAL_INVOCATION_X_ID,
        GLOBAL_INVOCATION_X_PTR_ID,

        (6 << 16) | OP_ACCESS_CHAIN,
        ELEMENT_POINTER_TYPE_ID,
        IN_ELEMENT_ID,
        IN_ID,
        CONSTANT_ZERO_ID,
        GLOBAL_INVOCATION_X_ID,

        (4 << 16) | OP_LOAD,
        INT_TYPE_ID,
        TEMP_LOADED_ID,
        IN_ELEMENT_ID,

        (6 << 16) | OP_ACCESS_CHAIN,
        ELEMENT_POINTER_TYPE_ID,
        OUT_ELEMENT_ID,
        OUT_ID,
        CONSTANT_ZERO_ID,
        GLOBAL_INVOCATION_X_ID,

        (3 << 16) | OP_STORE,
        OUT_ELEMENT_ID,
        TEMP_LOADED_ID,

        (1 << 16) | OP_RETURN,

        (1 << 16) | OP_FUNCTION_END,
    };
    const std::array shaderArray = std::to_array(shader);
    return shaderArray;
}