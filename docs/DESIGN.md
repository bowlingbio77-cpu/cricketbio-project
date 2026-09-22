---
name: Kinetic Biomechanics
colors:
  surface: '#f4f4f4'
  surface-dim: '#dadada'
  surface-bright: '#f9f9f9'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#edf1f5'
  surface-container: '#e5ebf2'
  surface-container-high: '#dbe3ec'
  surface-container-highest: '#cfd9e5'
  on-surface: '#0f2238'
  on-surface-variant: '#34495e'
  inverse-surface: '#2f3131'
  inverse-on-surface: '#f1f1f1'
  outline: '#8b9eb5'
  outline-variant: '#c4d1df'
  surface-tint: '#506070'
  primary: '#506070'
  on-primary: '#0c2543'
  primary-container: '#d9eafd'
  on-primary-container: '#596a7a'
  inverse-primary: '#b7c8db'
  secondary: '#406089'
  on-secondary: '#ffffff'
  secondary-container: '#aecefd'
  on-secondary-container: '#375880'
  tertiary: '#30647d'
  on-tertiary: '#ffffff'
  tertiary-container: '#ceecff'
  on-tertiary-container: '#3b6e87'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d3e4f7'
  primary-fixed-dim: '#b7c8db'
  on-primary-fixed: '#0c1d2a'
  on-primary-fixed-variant: '#384857'
  secondary-fixed: '#d3e3ff'
  secondary-fixed-dim: '#a9c9f7'
  on-secondary-fixed: '#001c38'
  on-secondary-fixed-variant: '#274870'
  tertiary-fixed: '#c1e8ff'
  tertiary-fixed-dim: '#9bcde9'
  on-tertiary-fixed: '#001e2b'
  on-tertiary-fixed-variant: '#124c64'
  background: '#f9f9f9'
  on-background: '#1a1c1c'
  surface-variant: '#e2e2e2'
  primary-contrast: '#0f2c52'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.025em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  title-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-xs:
    fontFamily: Inter
    fontSize: 10px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.06em
  metric-display:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.03em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.5rem
  gutter-mobile: 0.75rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style

The design system serves clinical sports science, high-performance telemetry, and biomechanical analysis. It unites analytical precision with elite athletic pacing. The user interface prioritizes frictionless data ingestion, real-time kinematic curves, vector trajectory displays, and gait symmetry diagnostics.

The visual direction follows a modern technical aesthetic: crisp structural geometry, high-density data tables, monospaced tabular figures, and deliberate contrast hierarchies. Visual styling reflects laboratory diagnostic monitors and motion telemetry consoles, avoiding extraneous decoration in favor of legible, high-contrast clarity.

## Colors

The system employs a high-contrast palette calibrated for light sports science environments and high-ambient-light diagnostic bays:

- **Neutral (`#f4f4f4`)**: The global surface foundation. Clean, low-glare, crisp sports science backdrop.
- **Primary (`#d9eafd`)**: Soft ice cobalt / pale cyan blue. Used as a high-visibility interactive surface, active highlight badge, and selected state surface. Because `#d9eafd` is a pale tint, content placed on it must use `on-primary` (`#0c2543`) to ensure strict WCAG AAA readability.
- **Secondary (`#1a3d64`)**: Deep navy steel. Anchors primary structural typography, solid buttons, primary navigation, frame contours, and high-emphasis data visualizations.
- **Tertiary (`#1d546c`)**: Teal slate / sports tech accent. Drives secondary telemetry curves, angle vectors, force plate vectors, and status annotations.
- **Surfaces & Containers**:
  - `surface-container-lowest` (`#ffffff`): Diagnostic cards, chart viewports, and table surfaces.
  - `surface-container-low` (`#edf1f5`): Alternating rows and filter docks.
  - `surface-container-high` (`#dbe3ec`): Inset tracks and hover states.
- **Typography on Surfaces**: `on-surface` (`#0f2238`) and `on-surface-variant` (`#34495e`) guarantee sharp optical contrast against all light neutral planes.

## Typography

Inter serves as the unified family across all typographic hierarchies. 

Numerical readout zones, telemetry indicators, force tables, and gait cycle metrics must enable tabular figures (`font-feature-settings: "tnum" 1`) to preserve column alignment across live data feeds. Technical markers, graph axis indicators, and anatomical tags use `label-xs` in full uppercase with expanded tracking for immediate identification.

## Layout & Spacing

Layout conforms to a responsive fluid grid optimized for high-density sensor readouts:
- **Desktop (`>= 1280px`)**: 12 columns, 24px (`1.5rem`) gutters, 32px (`2rem`) margins.
- **Tablet (`768px - 1279px`)**: 8 columns, 16px (`1rem`) gutters, 24px (`1.5rem`) margins.
- **Mobile (`< 768px`)**: 4 columns, 12px (`0.75rem`) gutters, 16px (`1rem`) margins.

Spacing follows an exact 4px/8px mathematical progression. Component padding adheres to `space-xs` (4px), `space-sm` (8px), `space-md` (16px), `space-lg` (24px), and `space-xl` (32px).

## Elevation & Depth

Visual hierarchy uses low-contrast technical outlines and disciplined tonal tiers:

- **Canvas Base**: Grounded on `#f4f4f4`.
- **Card Tier**: Rests on `#ffffff` bounded by 1px solid `#c4d1df`. Minimal shadow: `0 1px 3px rgba(26, 61, 100, 0.06), 0 1px 2px rgba(26, 61, 100, 0.04)`.
- **Active / Raised Elements**: Interactive telemetry panels elevate on hover with `0 4px 12px rgba(26, 61, 100, 0.10)` and outline `#8b9eb5`.
- **Floating Overlays & Menus**: Tooltips and inspector overlays utilize `#ffffff` with `0 12px 28px -4px rgba(26, 61, 100, 0.14)`, enclosed by 1px solid `#8b9eb5`.
- **Recessed Insets**: Waveform troughs and scrubbing tracks utilize `#e5ebf2` with an inner shadow: `inset 0 1px 2px rgba(26, 61, 100, 0.08)`.

## Shapes

The interface uses a calibrated soft contour (Level 1) to convey laboratory precision:
- Standard UI elements (buttons, segmented controls, table cells, text inputs): `0.25rem` (4px).
- Cards, telemetry containers, and charts: `rounded-lg` (`0.5rem` / 8px).
- Floating dialogs and modal panels: `rounded-xl` (`0.75rem` / 12px).
- Continuous sensor nodes, status pings, and joint point markers: `9999px` (fully circular).

## Components

### Buttons
- **Primary (High-Contrast Solid)**: Background `#1a3d64` (secondary deep navy steel), text `#ffffff`, border none. Hover: `#122c4a`. Active: `#0b1c30`. Focus outline: 2px solid `#1d546c`.
- **Accent / Highlight (Ice Cobalt)**: Background `#d9eafd`, text `#0c2543`, border 1px solid `#c4d1df`. Hover: `#c0dcfa`. Active: `#a8ccf5`.
- **Secondary (Outlined)**: Background `#ffffff`, text `#1a3d64`, border 1px solid `#c4d1df`. Hover: `#edf1f5`.
- **Tertiary / Ghost**: Background transparent, text `#1d546c`. Hover: `#e5ebf2`.

### Chips & Segmented Controls
- Height: 28px. Border radius: 4px. Typographic level: `label-md`.
- **Unselected**: Background `#ffffff`, text `#34495e`, border 1px solid `#c4d1df`.
- **Selected**: Background `#d9eafd`, text `#0c2543`, border 1px solid `#1a3d64`, font weight 600.

### Input Fields
- Background `#ffffff`, border 1px solid `#c4d1df`, text `#0f2238`, placeholder `#8b9eb5`. Focus state: border 1.5px solid `#1a3d64`, ring: 2px solid `#d9eafd`. Height: 36px (compact analytics view) or 44px (standard).

### Checkboxes & Radio Controls
- Base: 16px × 16px.
- **Unchecked**: Background `#ffffff`, border 1.5px solid `#8b9eb5`.
- **Checked**: Background `#1a3d64`, border `#1a3d64`, checkmark `#ffffff`. Radius: 4px for checkboxes, circular for radios.

### Cards & Telemetry Containers
- Background `#ffffff`, border 1px solid `#c4d1df`, border radius 8px, padding `space-md` (16px) or `space-lg` (24px).
- Headers pair an `on-surface` (`#0f2238`) label with a `#d9eafd` highlight pill or `#1d546c` metric delta indicator.

### Kinematic Data Tables
- Table header: Background `#edf1f5`, text `#34495e`, border-bottom 1px solid `#c4d1df`, `label-xs` uppercase.
- Rows: Background `#ffffff` alternating with `#f4f4f4`. Hover row: `#d9eafd` at 40% opacity. Numerical cells apply `font-feature-settings: "tnum" 1`.