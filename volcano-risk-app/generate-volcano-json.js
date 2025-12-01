// generate-volcano-json.js
// This script reads volcano_catalog_clean.csv and generates volcano_risk.json with correct column mappings

const fs = require('fs');
const path = require('path');
const Papa = require('papaparse');

const CSV_PATH = 'D:\\Riska\\Terramind\\volcano-risk-app\\data\\volcano_catalog_clean.csv';
const OUTPUT_PATH = 'D:\\Riska\\Terramind\\volcano-risk-app\\public\\data\\volcano_risk.json';

console.log('📖 Reading CSV file...');

// Read the CSV file
const csvData = fs.readFileSync(CSV_PATH, 'utf8');

console.log('🔄 Parsing CSV data...');

// Parse CSV with PapaParse
const parsed = Papa.parse(csvData, {
  header: true,
  dynamicTyping: true,
  skipEmptyLines: true,
  transformHeader: (header) => header.trim() // Remove any whitespace from headers
});

console.log(`✅ Found ${parsed.data.length} volcanoes in CSV`);
console.log('📊 Sample columns:', Object.keys(parsed.data[0] || {}).slice(0, 10));

// Print first valid row to debug
const firstRow = parsed.data.find(row => row.name && row.latitude && row.longitude);
if (firstRow) {
  console.log('\n🔍 First valid row sample:');
  console.log('  name:', firstRow.name);
  console.log('  country:', firstRow.country);
  console.log('  latitude:', firstRow.latitude);
  console.log('  longitude:', firstRow.longitude);
  console.log('  status_simple:', firstRow.status_simple);
  console.log('  last_eruption_year_num:', firstRow.last_eruption_year_num);
  console.log('');
}

const features = [];
let successfulAssessments = 0;
let skippedCount = 0;

parsed.data.forEach((row, index) => {
  // Skip rows without essential data - FIXED: use 'name' instead of 'volcano_name'
  if (!row.name || !row.latitude || !row.longitude) {
    skippedCount++;
    return; // Silent skip - don't log every skip
  }

  try {
    const feature = {
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [
          parseFloat(row.longitude) || 0,
          parseFloat(row.latitude) || 0
        ]
      },
      properties: {
        id: row.id || `volcano_${index}`,
        name: row.name || 'Unknown', // FIXED: use 'name' column
        country: row.country || 'Unknown',
        status: row.status || 'Unknown',
        status_simple: row.status_simple || 'Unknown',
        last_eruption_year: parseInt(row.last_eruption_year_num) || null, // Use correct column
        risk_score: parseFloat(row.risk_score) || 0,
        risk_category: row.risk_category || 'UNKNOWN',
        predicted_fatalities: parseFloat(row.predicted_fatalities) || 0,
        confidence_level: parseFloat(row.confidence_level) || 0,
        total_population: parseFloat(row.total_population) || 0,
        high_risk_population: parseFloat(row.high_risk_population) || 0,
        economic_loss_usd: parseFloat(row.economic_loss_usd) || 0,
        slope_mean: parseFloat(row.slope_mean) || 0,
        high_hazard_area_pct: parseFloat(row.high_hazard_area_pct) || 0,
        forest_at_risk_km2: parseFloat(row.forest_at_risk_km2) || 0,
        assessment_date: row.assessment_date || new Date().toISOString(),
        assessment_year: parseInt(row.assessment_year) || new Date().getFullYear(),
        buffer_radius_km: parseFloat(row.buffer_radius_km) || 10,
        has_dem_raster: row.has_dem_raster === true || row.has_dem_raster === 'true',
        has_lulc_raster: row.has_lulc_raster === true || row.has_lulc_raster === 'true'
      }
    };

    features.push(feature);
    successfulAssessments++;
  } catch (error) {
    console.error(`❌ Error processing row ${index + 1}:`, error.message);
  }
});

console.log(`\n⚠️  Skipped ${skippedCount} rows without essential data (name, lat, lng)`);
console.log(`✅ Successfully processed ${successfulAssessments} volcanoes`);

const geoJSON = {
  type: "FeatureCollection",
  metadata: {
    title: "Global Volcano Risk Assessment",
    generated_at: new Date().toISOString(),
    total_volcanoes: parsed.data.length,
    successful_assessments: successfulAssessments,
    source_csv: CSV_PATH,
    columns_mapped: {
      last_eruption_year: "last_eruption_year_num (from CSV)",
      status_simple: "status_simple (from CSV)",
      name: "name (from CSV)"
    }
  },
  features: features
};

console.log('\n📝 Writing JSON file...');

// Create directory if it doesn't exist
const outputDir = path.dirname(OUTPUT_PATH);
if (!fs.existsSync(outputDir)) {
  console.log(`📁 Creating directory: ${outputDir}`);
  fs.mkdirSync(outputDir, { recursive: true });
}

fs.writeFileSync(OUTPUT_PATH, JSON.stringify(geoJSON, null, 2), 'utf8');

console.log(`\n✅ SUCCESS! Generated volcano_risk.json`);
console.log(`📍 Location: ${OUTPUT_PATH}`);
console.log(`📊 Total volcanoes: ${successfulAssessments}`);

if (features.length > 0) {
  console.log('\n🔍 Sample volcano data:');
  console.log(JSON.stringify(features[0], null, 2));

  // Print statistics about last eruption years
  const withEruptionYear = features.filter(f => f.properties.last_eruption_year && f.properties.last_eruption_year > 0);
  console.log(`\n📅 Volcanoes with eruption year data: ${withEruptionYear.length} / ${successfulAssessments}`);

  // Print statistics about status
  const statusCounts = {};
  features.forEach(f => {
    const status = f.properties.status_simple || 'Unknown';
    statusCounts[status] = (statusCounts[status] || 0) + 1;
  });
  console.log('\n🌋 Activity Status Distribution:');
  Object.entries(statusCounts).forEach(([status, count]) => {
    console.log(`   ${status}: ${count}`);
  });
} else {
  console.log('\n⚠️  WARNING: No features were generated!');
  console.log('Please check your CSV file format.');
}