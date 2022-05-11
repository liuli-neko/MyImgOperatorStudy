add_requires("opencv")
add_requires("libjpeg")
add_requires("libpng")
add_requires("libwebp")
add_rules("mode.debug","mode.release")

add_cxxflags("-fexec-charset=GBK")
target("sift")
    add_includedirs("/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("*.cpp")
    if is_mode("debug") then
        add_defines("DEBUG")
    end