// merge-volcano-data.js
// This script merges last_eruption_year and other metadata from the old file into the new file

const fs = require('fs');
const path = require('path');

const OLD_FILE = 'D:\\Riska\\Terramind\\volcano-risk-app\\public\\data\\volcano_risk.json';
const NEW_FILE = 'D:\\Riska\\Terramind\\volcano-risk-web\\public\\data\\volcano_risk.json';
const OUTPUT_FILE = 'D:\\Riska\\Terramind\\volcano-risk-web\\public\\data\\volcano_risk.json';

console.log('🔄 Starting merge process...\n');

// Read both files
console.log('📖 Reading old file (with correct eruption years)...');
const oldData = JSON.parse(fs.readFileSync(OLD_FILE, 'utf8'));
console.log(`   ✅ Loaded ${oldData.features.length} volcanoes from old file`);

console.log('📖 Reading new file (with correct risk data)...');
const newData = JSON.parse(fs.readFileSync(NEW_FILE, 'utf8'));
console.log(`   ✅ Loaded ${newData.features.length} volcanoes from new file`);

// Create a map of old data by volcano name for quick lookup
const oldDataMap = {};
oldData.features.forEach(feature => {
  const name = feature.properties.name.trim().toLowerCase();
  oldDataMap[name] = feature.properties;
});

console.log('\n🔀 Merging data...');

let matchedCount = 0;
let unmatchedCount = 0;

// Merge data
newData.features.forEach(feature => {
  const name = feature.properties.name.trim().toLowerCase();

  if (oldDataMap[name]) {
    // Found a match - copy the missing fields
    const oldProps = oldDataMap[name];

    feature.properties.id = oldProps.id || feature.properties.id;
    feature.properties.country = oldProps.country || feature.properties.country;
    feature.properties.status = oldProps.status || feature.properties.status;
    feature.properties.status_simple = oldProps.status_simple || feature.properties.status_simple;
    feature.properties.last_eruption_year = oldProps.last_eruption_year || feature.properties.last_eruption_year;

    matchedCount++;
  } else {
    unmatchedCount++;
    console.log(`   ⚠️  No match found for: ${feature.properties.name}`);
  }
});

console.log(`\n📊 Merge Results:`);
console.log(`   ✅ Matched and merged: ${matchedCount} volcanoes`);
console.log(`   ⚠️  Unmatched: ${unmatchedCount} volcanoes`);

// Update metadata
newData.metadata.merged_at = new Date().toISOString();
newData.metadata.merge_source = OLD_FILE;

// Write the merged data
console.log('\n💾 Writing merged data to output file...');
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(newData, null, 2), 'utf8');

console.log(`\n✅ SUCCESS! Merged data written to:`);
console.log(`   ${OUTPUT_FILE}`);

// Show sample of merged data
console.log('\n📋 Sample merged volcano (Merapi):');
const merapi = newData.features.find(f => f.properties.name.toLowerCase() === 'merapi');
if (merapi) {
  console.log(JSON.stringify({
    name: merapi.properties.name,
    country: merapi.properties.country,
    status: merapi.properties.status,
    status_simple: merapi.properties.status_simple,
    last_eruption_year: merapi.properties.last_eruption_year,
    risk_score: merapi.properties.risk_score,
    risk_category: merapi.properties.risk_category,
    predicted_fatalities: merapi.properties.predicted_fatalities,
    total_population: merapi.properties.total_population
  }, null, 2));
} else {
  console.log('   ⚠️  Merapi not found in data');
}

console.log('\n🎉 Merge complete! You can now run: npm run dev');