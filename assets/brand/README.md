# FlashIM pixel kit

All sprites are drawn on an exact pixel grid. Scale PNGs only by whole numbers
(2x, 4x, 8x...) with nearest-neighbour. SVGs scale cleanly to any size.

- `svg/`: mark-pair (primary), mark-match-bolt, mark-monogram, favicon-16,
  wordmark-{bolt,match,ink}, lockup-{dark,light}
- `png/`: every sprite at @1x, @4x, @8x and @16x, plus app-icon-192/512
- `favicon.ico`: 16 px (favicon sprite) + 32 px (full mark)

README header that switches with GitHub's light/dark theme:

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/lockup-dark@8x.png">
    <img alt="FlashIM" src="assets/lockup-light@8x.png" width="560">
  </picture>
</p>
```

Docs site favicon:

```html
<link rel="icon" href="favicon.ico" sizes="16x16 32x32">
<link rel="icon" type="image/svg+xml" href="svg/favicon-16.svg">
```

Palette: Ink #0B0D1A, Night #171B2E, Slate #262C47, Dusk #3B4468, Haze #6E7AA6,
Paper #F4F1E6, Bolt #FFD23F, Spark #FFF3B0, Ember #F08A0C, Match #3FE0F2,
Keypoint #FF5C8D, Ridge #7FE3A0, Moss #3E9A6B
