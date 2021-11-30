project('ries', 'c',
  version : '2018-08-02',
  default_options : ['warning_level=3'])

executable('ries',
           'ries.c',
           install: true,
           dependencies: meson.get_compiler('c').find_library('m', required : false))

install_man('ries.1')
