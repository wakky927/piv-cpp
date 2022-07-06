add_library(piv_lib
        SHARED
            module/KC.cpp
        )

target_include_directories(piv_lib
        PUBLIC include
        )

add_executable(piv_cpp main.cpp)
