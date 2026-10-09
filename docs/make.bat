@ECHO OFF
set SPHINXBUILD=sphinx-build
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% -W --keep-going -b %1 %SOURCEDIR% %BUILDDIR%/%1
goto end

:help
echo Usage: make.bat html ^| doctest ^| linkcheck

:end
