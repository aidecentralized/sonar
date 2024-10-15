# Common Errors during installation

### ImportError: MagickWand shared library not found.

This issue comes up for mac users in particular:
Solution:
```
brew install imagemagick@6
export MAGICK_HOME=/opt/homebrew/opt/imagemagick@6
```