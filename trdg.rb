require "fileutils"

font_dir_path = File.join(__dir__, "fonts/*/font/*.ttf")

Dir[font_dir_path].each do |ttf_path|
  font_id = ttf_path[/fonts\/(\d+)\/font/, 1]
  font_name = ttf_path.split("/").last[/(.+).ttf/, 1]

  # Delete all default images
  FileUtils.rm_rf(Dir[File.join(__dir__, "fonts/#{font_id}/data_set/*")])
  
  # Run TRDG Docker command
  puts File.join(__dir__)
  params = {
    language: "ko",
    count: 2350,
    format: 48,
    skew_angle: 1,
    random_skew: "",
    background: 0, # Gausian Noise
    # blur: 1,
    font: "fonts/#{font_id}/font/#{font_name}.ttf",
    input_file: "dicts/ko.txt",
    output_dir: "fonts/#{font_id}/data_set",
  }

  puts `docker run -v #{File.join(__dir__)}/:/app -t belval/trdg:latest trdg #{params.map { |k, v| "--#{k} #{v}".strip }.join(" ")}`
end