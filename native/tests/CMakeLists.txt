add_executable(test_golden test_golden.cpp)
target_link_libraries(test_golden PRIVATE sstvae_core)
target_include_directories(test_golden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The corpus path is passed in rather than compiled in, so the binary
# can be pointed at a regenerated corpus without a rebuild while a
# format change is being reviewed.
add_test(NAME golden
         COMMAND test_golden ${CMAKE_CURRENT_SOURCE_DIR}/golden
                 ${CMAKE_CURRENT_SOURCE_DIR}/../../sstvae/modem/interleaver_perms.npy)

add_executable(test_npy test_npy.cpp)
target_link_libraries(test_npy PRIVATE sstvae_core)
target_include_directories(test_npy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME npy COMMAND test_npy ${CMAKE_CURRENT_SOURCE_DIR}/golden)
