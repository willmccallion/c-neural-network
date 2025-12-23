cmake_minimum_required(VERSION 3.24)
project(draw_predictor C)

set(CMAKE_C_STANDARD 11)
cmake_policy(SET CMP0135 NEW)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -ffast-math -funroll-loops)
endif()

# Get raylib
include(FetchContent)
FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/archive/master.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(raylib)

include_directories(include)

set(SOURCES
    src/main.c
    src/nn.c
    src/visualizer.c
    src/mnist.c
    src/utils.c
    src/image_proc.c
    src/gui.c
    src/trainer.c
)

add_executable(${PROJECT_NAME} ${SOURCES})

# Define paths
set(DATA_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/data.py")
set(DATA_OUTPUT "${CMAKE_BINARY_DIR}/data/extended-train-images-idx3-ubyte")
set(VENV_DIR "${CMAKE_BINARY_DIR}/venv")

# Determine OS-specific Venv paths
if(WIN32)
    set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
    set(VENV_PIP "${VENV_DIR}/Scripts/pip.exe")
else()
    set(VENV_PYTHON "${VENV_DIR}/bin/python")
    set(VENV_PIP "${VENV_DIR}/bin/pip")
endif()

# Create a custom command to generate the data
add_custom_command(
    OUTPUT ${DATA_OUTPUT}

    # Create virtual environment 
    COMMAND ${Python3_EXECUTABLE} -m venv ${VENV_DIR}

    # Install dependencies 
    COMMAND ${VENV_PIP} install --disable-pip-version-check --quiet numpy requests tqdm

    # Run the data script
    COMMAND ${CMAKE_COMMAND} -E echo "Download datasets (This only happens once.)"
    COMMAND ${VENV_PYTHON} ${DATA_SCRIPT}

    # Run this from the build directory so 'data/' is created inside build/
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}

    COMMENT "Checking for training data..."
    VERBATIM
)

# Create a target that depends on the output file
add_custom_target(generate_training_data DEPENDS ${DATA_OUTPUT})

# Make the main executable depend on the data generation
add_dependencies(${PROJECT_NAME} generate_training_data)

target_link_libraries(${PROJECT_NAME} raylib m OpenMP::OpenMP_C)
