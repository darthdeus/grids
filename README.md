# `grids` Crate

This crate provides a simple and flexible 2D grid data structure mainly intended for grid based games.

If you need a _matrix_ this is not the right crate.

## Features

- Flexible grid creation.
- Cloning sub-sections of the grid based on rectangular bounds.
- Pasting one grid into another at a specified offset.
- Many convenient iterators with/without coordinates in various types.
- Clamped access methods to ensure coordinates remain within grid boundaries.
- Support for serialization and deserialization via `serde` feature.
- Uses `glam` and allows indexing with `IVec2/UVec2` for extra convenience.

## Example Usage

Here's a simple example showcasing the usage of the crate:

```rust
let mut grid = Grid::new(3, 2, 0); // A 3x2 grid filled with zeros.
grid[(0, 1)] = 5;

// Accessing using glam::IVec2.
assert_eq!(grid[glam::IVec2::new(1, 0)], 0);
// Accessing using glam::UVec2.
assert_eq!(grid[glam::UVec2::new(0, 1)], 5);

// Converting grid to a Vec.
assert_eq!(
    grid.into_iter_values().collect::<Vec<_>>(),
    vec![0, 0, 0, 5, 0, 0]
);
```
