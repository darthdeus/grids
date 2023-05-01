use glam::IVec2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Grid<T> {
    data: Vec<T>,
    width: i32,
    height: i32,
}

impl<T: Clone> Grid<T> {
    pub fn new(width: i32, height: i32, value: T) -> Self {
        Self {
            data: vec![value; (width * height) as usize],
            width,
            height,
        }
    }

    pub fn filled_with<F: FnMut(i32, i32) -> T>(width: i32, height: i32, mut f: F) -> Self {
        let mut data = Vec::with_capacity((width * height) as usize);

        for x in 0..width {
            for y in 0..height {
                data.push(f(x, y));
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    pub fn width(&self) -> i32 {
        self.width as i32
    }

    pub fn height(&self) -> i32 {
        self.height as i32
    }

    pub fn is_valid(&self, coord: glam::IVec2) -> bool {
        coord.x >= 0 && coord.x < self.width() && coord.y >= 0 && coord.y < self.height()
    }

    pub fn get(&self, x: i32, y: i32) -> &T {
        &self[(x, y)]
    }

    pub fn get_mut(&mut self, x: i32, y: i32) -> &mut T {
        &mut self[(x, y)]
    }

    pub fn get_clamped(&self, x: i32, y: i32) -> &T {
        let x = x.clamp(0, self.width as i32 - 1);
        let y = y.clamp(0, self.height as i32 - 1);

        self.get(x, y)
    }

    pub fn v_clamped(&self, v: glam::Vec2) -> &T {
        let x = v.x as i32;
        let y = v.y as i32;

        let x = x.clamp(0, self.width as i32 - 1);
        let y = y.clamp(0, self.height as i32 - 1);

        self.get(x, y)
    }

    pub fn get_clamped_mut(&mut self, x: i32, y: i32) -> &mut T {
        let x = x.clamp(0, self.width as i32 - 1);
        let y = y.clamp(0, self.height as i32 - 1);

        self.get_mut(x, y)
    }

    pub fn iter_rect(&self, min: IVec2, max: IVec2) -> impl Iterator<Item = (i32, i32, &T)> {
        let mut coords = vec![];

        for x in min.x..max.x {
            for y in min.y..max.y {
                coords.push((x, y));
            }
        }

        coords.into_iter().map(|(x, y)| (x, y, &self[(x, y)]))
    }

    pub fn clone_rect(&self, min: IVec2, max: IVec2) -> Grid<T> {
        let dims = max - min;
        let mut result = Grid::new(dims.x, dims.y, self[(0i32, 0i32)].clone());

        for x in 0..dims.x {
            for y in 0..dims.y {
                result[(x, y)] = self[(x + min.x, y + min.y)].clone();
            }
        }

        result
    }

    pub fn paste_grid(&mut self, offset: IVec2, other: &Grid<T>) {
        for x in 0..other.width() {
            for y in 0..other.height() {
                self[(offset.x + x, offset.y + y)] = other[(x, y)].clone();
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (i32, i32, &T)> {
        self.data.iter().enumerate().map(|(i, v)| {
            let i = i as i32;
            let x = i % self.width as i32;
            let y = i / self.width as i32;

            (x, y, v)
        })
    }

    pub fn iter_values(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn into_iter_values(self) -> impl Iterator<Item = T> {
        self.data.into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (i32, i32, &mut T)> {
        self.data.iter_mut().enumerate().map(|(i, v)| {
            let i = i as i32;
            let x = i % self.width as i32;
            let y = i / self.height as i32;

            (x, y, v)
        })
    }

    pub fn iter_values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    pub fn iter_coords(&self) -> impl Iterator<Item = (glam::IVec2, &T)> {
        self.iter()
            .map(|(x, y, v)| (glam::IVec2::new(x as i32, y as i32), v))
    }

    pub fn iter_coords_mut(&mut self) -> impl Iterator<Item = (glam::IVec2, &mut T)> {
        self.iter_mut().map(|(x, y, v)| (glam::IVec2::new(x, y), v))
    }

    pub fn coords(&self) -> Vec<glam::IVec2> {
        self.iter()
            .map(|(x, y, _)| glam::IVec2::new(x, y))
            .collect::<Vec<_>>()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn size(&self) -> IVec2 {
        IVec2::new(self.width(), self.height())
    }

    pub fn pack(fill: T, grids: Vec<Grid<T>>) -> Grid<T> {
        assert!(grids.len() > 0);

        let mut size = IVec2::ZERO;

        for grid in grids.iter() {
            size.x += grid.width();
            size.y = size.y.max(grid.height());
        }

        let mut result = Grid::new(size.x, size.y, fill.clone());

        let mut offset = IVec2::ZERO;

        for grid in grids.iter() {
            for (x, y, val) in grid.iter() {
                result[offset + IVec2::new(x, y)] = val.clone();
            }

            offset += IVec2::new(grid.width(), 0);
        }

        result
    }
}

impl<T: Clone> Index<(i32, i32)> for Grid<T> {
    type Output = T;

    fn index(&self, (x, y): (i32, i32)) -> &Self::Output {
        &self.data[(x + y * self.width) as usize]
    }
}

impl<T: Clone> IndexMut<(i32, i32)> for Grid<T> {
    fn index_mut(&mut self, (x, y): (i32, i32)) -> &mut Self::Output {
        &mut self.data[(x + y * self.width) as usize]
    }
}

// impl<T: Clone> Index<(u32, u32)> for Grid<T> {
//     type Output = T;
//
//     fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
//         &self.data[(x as i32 + y as i32 * self.width) as usize]
//     }
// }
//
// impl<T: Clone> IndexMut<(u32, u32)> for Grid<T> {
//     fn index_mut(&mut self, (x, y): (u32, u32)) -> &mut Self::Output {
//         &mut self.data[(x as i32 + y as i32 * self.width) as usize]
//     }
// }

impl<T: Clone> Index<glam::IVec2> for Grid<T> {
    type Output = T;

    fn index(&self, index: glam::IVec2) -> &Self::Output {
        &self[(index.x, index.y)]
    }
}

impl<T: Clone> IndexMut<glam::IVec2> for Grid<T> {
    fn index_mut(&mut self, index: glam::IVec2) -> &mut Self::Output {
        &mut self[(index.x, index.y)]
    }
}

impl<T: Clone> Index<glam::UVec2> for Grid<T> {
    type Output = T;

    fn index(&self, index: glam::UVec2) -> &Self::Output {
        &self[(index.x as i32, index.y as i32)]
    }
}

impl<T: Clone> IndexMut<glam::UVec2> for Grid<T> {
    fn index_mut(&mut self, index: glam::UVec2) -> &mut Self::Output {
        &mut self[(index.x as i32, index.y as i32)]
    }
}

#[test]
fn test_stuff() {
    let mut grid = Grid::new(3, 2, 0);
    grid[(0, 1)] = 5;

    assert_eq!(grid[glam::IVec2::new(1, 0)], 0);
    assert_eq!(grid[glam::UVec2::new(0, 1)], 5);

    assert_eq!(
        grid.into_iter_values().collect::<Vec<_>>(),
        vec![0, 0, 0, 5, 0, 0]
    );
}
