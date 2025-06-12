# CWE-22: Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')

Description
    





The product uses external input to construct a pathname that is intended to identify a file or directory that is located underneath a restricted parent directory, but the product does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location that is outside of the restricted directory.