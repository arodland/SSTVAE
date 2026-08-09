plugins {
    id("com.android.application")
}

android {
    namespace = "org.cleverdomain.sstvae.smoke"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "org.cleverdomain.sstvae.smoke"
        // Qt 6.8 sets the Tier 0 floor at 28; matched here so the smoke
        // test is not accidentally proving things on an API level the
        // real app will not support. Nothing this app uses needs more:
        // getDevices() and setPreferredDevice() are both API 23.
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"

        externalNativeBuild {
            cmake {
                // The core is C++20 and compares against Python to 1e-12,
                // so fast-math is out for the same reason it is out on the
                // desktop: it would licence reassociating exactly the sums
                // those tolerances measure.
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                )
                cppFlags += listOf("-fno-fast-math")
                // Without this AGP builds every target the CMake tree
                // defines -- which here means the desktop CLI tools and
                // the whole ctest suite, cross-compiled for the phone
                // and then discarded. EXCLUDE_FROM_ALL on the
                // subdirectory does not help, because AGP asks for the
                // targets by name rather than building `all`.
                targets += "sstvae_smoke"
            }
        }

        // arm64 for real phones, x86_64 for the emulator. Deliberately
        // not armeabi-v7a: 32-bit ARM is a shrinking population and every
        // extra ABI is another 10 MB of onnxruntime in the APK.
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // The onnxruntime .so comes from the AAR that CMake downloads and is
    // placed by the native build, so nothing extra is packaged here.
    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
}

dependencies {
    // appcompat drags in kotlin-stdlib-jdk7/jdk8 1.6.21 alongside
    // kotlin-stdlib 1.8.22, and since 1.8 the jdk7/jdk8 artifacts were
    // folded into the main one -- so the same classes arrive twice and
    // dexing refuses. The BOM aligns all three, which is upstream's own
    // answer; excluding the jdk artifacts also works but silently drops
    // whatever else they might carry.
    implementation(platform("org.jetbrains.kotlin:kotlin-bom:1.8.22"))
    implementation("androidx.appcompat:appcompat:1.7.0")
}
