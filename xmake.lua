add_requires("opencv")
add_requires("libjpeg")
add_requires("libpng")
add_requires("libwebp")
add_rules("mode.debug","mode.release")

add_includedirs("D:\\APP\\eigen-3.4.0")

target("main")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("img_io/*.cc")
    add_files("*.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("mat_test")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_files("test/mat_test.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("path_test")
    add_files("test/path_test.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("img_gauss_blur")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("img_io/*.cc")
    add_files("test/img_gauss_blur.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("img_histogram_equalization")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("img_io/*.cc")
    add_files("test/img_histogram_equalization.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end
target("img_dft")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("img_io/*.cc")
    add_files("test/img_dft.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("img_filter")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("img_io/*.cc")
    add_files("test/img_filter.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("math_test")
    add_files("test/math_test.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("main")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("img_io/*.cc")
    add_files("*.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end