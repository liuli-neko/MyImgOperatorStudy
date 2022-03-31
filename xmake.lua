add_requires("opencv")
add_requires("libjpeg")
add_requires("libpng")
add_requires("libwebp")
add_rules("mode.debug","mode.release")

target("main")
    add_includedirs("img_io/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_includedirs("D:/APP/eigen/eigen")
    add_includedirs("D:/APP/eigen/eigen/Eigen")
    add_files("img_io/*.cc")
    add_files("*.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end

target("mat_test")
    add_includedirs("D:/APP/eigen/eigen")
    add_includedirs("D:/APP/eigen/eigen/Eigen")
    add_includedirs("include")
    add_packages("opencv")
    add_files("test/mat_test.cc")
    if is_mode("debug") then 
        add_defines("DEBUG")
    end