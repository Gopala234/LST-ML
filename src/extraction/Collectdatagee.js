/**
 * GEE Script: Landsat 9 Monthly GeoTIFF Exporter
 *
 * This script processes Landsat 9 Level 2 data for a specific Area of Interest (AOI)
 * for a two-year period (2022-2023).
 *
 * It calculates NDVI, NDBI, NDWI, SAVI, Albedo, and LST for every clear pixel.
 * It then creates a single median composite (a single GeoTIFF) for each month.
 *
 * This results in 24 GeoTIFF export tasks (12 months * 2 years).
 */

// --- 1. DEFINE YOUR AREA OF INTEREST (AOI) ---
// IMPORTANT: Replace this placeholder with your actual AOI.
// Center the map on the AOI
Map.centerObject(aoi, 10);

// Add the AOI to the map as a visual layer
Map.addLayer(aoi, {color: 'FF0000'}, 'Area of Interest (AOI)');

// --- 2. SET PARAMETERS ---
var fullStartDate = ee.Date('2022-01-01');
var fullEndDate = ee.Date('2023-12-31'); // Set to end of 2023
var l9 = 'LANDSAT/LC09/C02/T1_L2';
var exportFolder = 'GEE_L9_Monthly_GeoTIFFs'; // New folder for monthly exports

// --- 3. HELPER FUNCTIONS (Unchanged) ---

/**
 * Applies scaling factors to Landsat 8/9 Level 2 data.
 */
function applyScaleFactors(image) {
  // Scale optical bands
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  // Scale thermal band (ST_B10) to Celsius
  var thermalBand = image.select('ST_B10').multiply(0.00341802).add(149.0)
                         .subtract(273.15) // Convert Kelvin to Celsius
                         .rename('LST_C');
  // Add QA band back
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true);
}

/**
 * Masks clouds and cloud shadows in Landsat 8/9 Level 2 data
 * using the QA_PIXEL band.
 */
function cloudMask(image) {
  var qa = image.select('QA_PIXEL');
  // Bits: 1 (Dilated Cloud), 3 (Cloud), 4 (Cloud Shadow)
  var dilatedCloud = 1 << 1;
  var cloud = 1 << 3;
  var cloudShadow = 1 << 4;

  // Mask is clear if all specified bits are 0
  var mask = qa.bitwiseAnd(dilatedCloud).eq(0)
               .and(qa.bitwiseAnd(cloud).eq(0))
               .and(qa.bitwiseAnd(cloudShadow).eq(0));

  return image.updateMask(mask);
}

/**
 * Main function to calculate all indices.
 * NOTE: We remove lon/lat bands as they don't work well with .median()
 * The GeoTIFF is inherently georeferenced, so they are not needed.
 */
function processImage(image) {
  // 1. Apply cloud mask FIRST (uses original QA band)
  var masked = cloudMask(image);

  // 2. Apply scaling factors to the masked image
  var scaled = applyScaleFactors(masked);

  // 3. Calculate Indices
  var ndvi = scaled.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var ndbi = scaled.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
  var ndwi = scaled.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI'); // McFeeters' NDWI

  var savi = scaled.expression(
    '((NIR - RED) / (NIR + RED + L)) * (1 + L)', {
      'NIR': scaled.select('SR_B5'),
      'RED': scaled.select('SR_B4'),
      'L': 0.5 // Soil adjustment factor
    }).rename('SAVI');

  // Albedo (Liang, 2001 formula adapted for L8/9 SR bands)
  var albedo = scaled.expression(
    '(0.356 * BLUE) + (0.130 * GREEN) + (0.373 * RED) + (0.085 * NIR) + (0.072 * SWIR1) + (0.0018 * SWIR2)', {
      'BLUE': scaled.select('SR_B2'),
      'GREEN': scaled.select('SR_B3'),
      'RED': scaled.select('SR_B4'),
      'NIR': scaled.select('SR_B5'),
      'SWIR1': scaled.select('SR_B6'),
      'SWIR2': scaled.select('SR_B7')
    }).rename('Albedo');

  // 4. Add LST (already calculated as 'LST_C' in applyScaleFactors)
  var lst = scaled.select('LST_C');

  // 5. Combine all bands for export
  var finalImage = ee.Image.cat([
    ndvi,
    ndbi,
    ndwi,
    savi,
    albedo,
    lst
  ]).toFloat(); // Convert to float for export compatibility

  // 7. Copy properties (like date) from original image
  return finalImage.copyProperties(image, ['system:time_start']);
}

// --- 4. LOAD AND PROCESS THE FULL COLLECTION ---

// Load the Landsat 9 collection for the entire date range
var collection = ee.ImageCollection(l9)
  .filterBounds(aoi)
  .filterDate(fullStartDate, fullEndDate);

// Map the processing function over the entire collection
var processedCollection = collection.map(processImage);

// --- 5. PREPARE MONTHLY COMPOSITE EXPORTS ---

// We will loop 24 times (12 months * 2 years)
var nMonths = fullEndDate.difference(fullStartDate, 'month').ceil();
print('Number of months to process:', nMonths);

// This is a client-side loop to create the tasks
for (var i = 0; i < nMonths.getInfo(); i++) {
  // Calculate the start and end for the current month
  var monthStartDate = fullStartDate.advance(i, 'month');
  var monthEndDate = monthStartDate.advance(1, 'month');

  // Filter the processed collection to just this month
  var monthCollection = processedCollection.filterDate(monthStartDate, monthEndDate);

  // Create a median composite
  // This takes all clear pixels from all images in the month
  // and finds the median value for each band.
  var monthMedian = monthCollection.median();

  // Get the month and year for the filename
  var baseName = monthStartDate.format('YYYY-MM').getInfo(); // <-- FIX: Added .getInfo()
  var fileName = 'L9_median_' + baseName;

  print('Preparing export for:', fileName);

  // --- Add LST layer for the first month (i=0) ---
  if (i === 0) {
    var lstVizParams = {
      min: 10, // Min LST in Celsius
      max: 35, // Max LST in Celsius
      palette: [
        '040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
        '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
        '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
        'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
        'ff2e02', 'ff0c00', 'ff0000', 'c20000', 'a50000', '790000'
      ]
    };
    Map.addLayer(monthMedian.select('LST_C').clip(aoi), lstVizParams, 'LST (Celsius) - ' + baseName);
    print('Added LST layer to map for:', baseName);
  }

  // --- Export GeoTIFF Task ---
  Export.image.toDrive({
    image: monthMedian.clip(aoi), // Clip to AOI to keep file size down
    description: fileName,
    folder: exportFolder,
    fileNamePrefix: fileName,
    region: aoi,
    scale: 30, // Native Landsat scale
    crs: 'EPSG:4326', // Standard Lon/Lat
    maxPixels: 1e13
  });
}



