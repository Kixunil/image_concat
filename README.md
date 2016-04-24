Image concat
============

Concatenate image tiles.

About
-----

This program concatenates images like `montage -mode concatenate` from Imagemagick package does but it is more intelligent in detecting intentions of the user. I came up with it when I got frustrated over misplaced tiles when invoking `montage` (which coordinate is which and how big the result should be). This also serves as a way to learn Rust for me.

Since this program is written in Rust, it should be guaranteed not to segfault and not to corrupt memory (hopefully also preventing stack overflow exploits), which is nice feature too.

Warning
-------

The code is far from perfect. I believe that at least ordering detection could be improved a little. It should be useful but there is no guarantee it will work for you! All error messages are just panics (but there is no sensible way of handling them other than exiting anyway).

Author
------

Martin Habovštiak
