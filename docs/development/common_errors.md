# Common Errors during installation

### ImportError: MagickWand shared library not found.

Solution:
```
brew install imagemagick@6
export MAGICK_HOME=/opt/homebrew/opt/imagemagick@6
```