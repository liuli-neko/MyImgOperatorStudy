add_rules("mode.debug","mode.release")

add_requires("libwebp", "libpng", "libjpeg", "opencv")
add_requires("eigen")
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

if is_plat("windows") then
  add_cxxflags("/utf-8")
end

if is_mode("debug") then
  add_defines("DEBUG")
end

add_packages("eigen")

target("main")
    add_includedirs("unit/")
    add_packages("libwebp", "libpng", "libjpeg", "opencv")
    add_files("unit/*.cc")
    add_files("*.cc")

target("mat_test")
    add_includedirs("unit/")
    add_packages("opencv")
    add_files("test/mat_test.cc")

target("path_test")
    add_files("test/path_test.cc")

target("img_gauss_blur")
    add_includedirs("unit/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/img_gauss_blur.cc")

target("img_histogram_equalization")
    add_includedirs("unit/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/img_histogram_equalization.cc")

target("img_dft")
    add_includedirs("unit/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/img_dft.cc")

target("img_filter")
    add_includedirs("unit/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/img_filter.cc")

target("math_test")
    add_files("test/math_test.cc")

target("img_morphology")
    add_includedirs("unit/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/img_morphology.cc")

target("img_sift")
    add_includedirs("unit/")
     add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/img_sift.cc")

target("kdtree_test")
    add_includedirs("unit/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/kdtree_test.cc")

target("img_detection_keypoint")
    add_includedirs("unit/")
    add_packages("opencv")
    add_packages("libjpeg")
    add_packages("libpng")
    add_packages("libwebp")
    add_files("unit/*.cc")
    add_files("test/img_detection_keypoint.cc")