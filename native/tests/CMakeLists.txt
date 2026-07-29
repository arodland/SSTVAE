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

# Corpus-free: the C++ modem against itself, so `ctest` covers the whole
# transmit/receive chain even where the Python reference is unavailable.
add_executable(test_modem_roundtrip test_modem_roundtrip.cpp)
target_link_libraries(test_modem_roundtrip PRIVATE sstvae_core)
target_include_directories(test_modem_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME modem_roundtrip COMMAND test_modem_roundtrip)

add_executable(test_ringbuffer test_ringbuffer.cpp)
target_link_libraries(test_ringbuffer PRIVATE sstvae_core)
target_include_directories(test_ringbuffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME ringbuffer COMMAND test_ringbuffer)

# The decode loop's decisions, with a stub decoder in place of the
# codec -- so the state machine is covered in every build, including
# --no-codec.
add_executable(test_rx_engine test_rx_engine.cpp)
target_link_libraries(test_rx_engine PRIVATE sstvae_core)
target_include_directories(test_rx_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME rx_engine COMMAND test_rx_engine)
