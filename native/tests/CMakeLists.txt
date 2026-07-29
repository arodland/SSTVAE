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

# The transmit sequence, with stub encoder/player/PTT -- so the "PTT
# always comes back down" guarantee is checked without a radio.
add_executable(test_tx_engine test_tx_engine.cpp)
target_link_libraries(test_tx_engine PRIVATE sstvae_core)
target_include_directories(test_tx_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME tx_engine COMMAND test_tx_engine)

# The device-independent half of the audio layer: rate conversion,
# sample-format conversion, device matching. No soundcard, no Qt.
add_executable(test_audio test_audio.cpp)
target_link_libraries(test_audio PRIVATE sstvae_core)
target_include_directories(test_audio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME audio COMMAND test_audio)

# Finding the model artifacts: names, cache, and --model resolution.
# No network -- the download is checked by running the CLI for real.
add_executable(test_checkpoint test_checkpoint.cpp)
target_link_libraries(test_checkpoint PRIVATE sstvae_core)
target_include_directories(test_checkpoint PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME checkpoint COMMAND test_checkpoint)

# Rig control's threading properties, against a fake radio -- so the
# part that can actually be wrong is covered with no libhamlib present.
add_executable(test_rig test_rig.cpp)
target_link_libraries(test_rig PRIVATE sstvae_core)
target_include_directories(test_rig PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME rig COMMAND test_rig)

# The libhamlib backend itself, exercised against Hamlib's own dummy
# rig (model 1) -- the same trick CLAUDE.md uses for the Python side,
# and the only way to run open/PTT/frequency with no radio attached.
if(TARGET sstvae_rig)
  add_executable(test_rig_hamlib test_rig_hamlib.cpp)
  target_link_libraries(test_rig_hamlib PRIVATE sstvae_core sstvae_rig)
  target_include_directories(test_rig_hamlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME rig_hamlib COMMAND test_rig_hamlib)
endif()

# Hard upper bounds, so a wedged test is reported as a Timeout rather
# than running until GitHub kills the whole job -- at which point there
# is no ctest output at all to say which test it was.
#
# Sized against the sanitizer build, which is the slowest thing that
# runs these: rx_engine measures ~150 s there and tx_engine ~26 s, so
# these are roughly 2.5x headroom. They bound the job where the
# per-wait watchdog inside test_rx_engine.cpp cannot -- that one gives
# the better message, but several waits expiring in turn would still
# outlast the run.
set_tests_properties(rx_engine PROPERTIES TIMEOUT 360)
set_tests_properties(tx_engine PROPERTIES TIMEOUT 180)
set_tests_properties(golden npy modem_roundtrip ringbuffer audio rig
                     checkpoint
                     PROPERTIES TIMEOUT 300)
if(TARGET sstvae_rig)
  set_tests_properties(rig_hamlib PROPERTIES TIMEOUT 300)
endif()
