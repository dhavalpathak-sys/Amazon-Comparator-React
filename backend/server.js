const express = require('express');
const path = require('path');
const cors = require('cors');
const axios = require('axios');
const { parse } = require('csv-parse/sync');
const { spawn } = require('child_process');
const { google } = require('googleapis');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000; 
// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use(express.urlencoded({ extended: true }));

// Your Google Sheet CSV URL
const SHEET_CSV_URL = 
`https://docs.google.com/spreadsheets/d/1C6s96_hmDTYjwwkq5r4ekAh5FrtAvPJqN54WwYl_eVI/export?format=csv&gid=75817034`;

// Golden URL Sheet CSV URL (Page 2)
const GOLDEN_SHEET_CSV_URL = 
`https://docs.google.com/spreadsheets/d/1C6s96_hmDTYjwwkq5r4ekAh5FrtAvPJqN54WwYl_eVI/export?format=csv&gid=62256398`;

// Cache for sheet data
let sheetDataCache = null;
let lastCacheTime = null;

// Cache for golden sheet data
let goldenSheetDataCache = null;
let lastGoldenCacheTime = null;

// Function to get real data from Google Sheets CSV
async function getRealSheetData() {
    // Return cached data if it's fresh (5 minutes)
    if (sheetDataCache && lastCacheTime && (Date.now() - lastCacheTime) < 3000000) {
        console.log('📊 Using cached sheet data');
        return sheetDataCache;
    }

    try {
        console.log('📊 Fetching fresh data from Google Sheets CSV...');
        const response = await axios.get(SHEET_CSV_URL, { 
            timeout: 15000,
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        });
        
        const csvData = response.data;
        
        // Debug: Log first 200 characters to see what we're receiving
        console.log('📝 First 200 chars of response:', csvData.substring(0, 200));
        console.log('📝 Response type:', typeof csvData);
        
        // Parse CSV
        const records = parse(csvData, {
            columns: true,
            skip_empty_lines: true,
            trim: true,
            relax_quotes: true,
            relax_column_count: true
        });

        console.log('✅ Successfully parsed', records.length, 'rows from Google Sheets');
        
        // Cache the data
        sheetDataCache = {
            success: true,
            data: records,
            headers: Object.keys(records[0] || {})
        };
        lastCacheTime = Date.now();
        
        return sheetDataCache;
        
    } catch (error) {
        console.error('❌ Error fetching from Google Sheets CSV:', error.message);
        
        // If it's an axios error, log the response
        if (error.response) {
            console.error('📍 Status:', error.response.status);
            console.error('📍 Response preview:', error.response.data?.substring(0, 200));
        }
        
        // Return comprehensive fallback data
        return {
            success: true,
            data: [
                { '': 'Clothing', Gender: 'Women', 'Age Group': 'Adults', Subcategory: 'Kurtas', URL: 'https://www.amazon.in/s?k=women+kurtas' },
                { '': 'Clothing', Gender: 'Women', 'Age Group': 'Adults', Subcategory: 'Sarees', URL: 'https://www.amazon.in/s?k=women+sarees' },
                { '': 'Clothing', Gender: 'Women', 'Age Group': 'Adults', Subcategory: 'Dresses', URL: 'https://www.amazon.in/s?k=women+dresses' },
                { '': 'Clothing', Gender: 'Men', 'Age Group': 'Adults', Subcategory: 'Shirts', URL: 'https://www.amazon.in/s?k=men+shirts' },
                { '': 'Clothing', Gender: 'Men', 'Age Group': 'Adults', Subcategory: 'T-Shirts', URL: 'https://www.amazon.in/s?k=men+t+shirts' }
            ],
            headers: ['', 'Gender', 'Age Group', 'Subcategory', 'URL'],
            fromCache: true
        };
    }
}

// Function to get golden sheet data (Page 2)
async function getGoldenSheetData() {
    // Return cached data if it's fresh (5 minutes)
    // if (goldenSheetDataCache && lastGoldenCacheTime && (Date.now() - lastGoldenCacheTime) < 3000000) {
    //     console.log('📊 Using cached golden sheet data');
    //     return goldenSheetDataCache;
    // }

    try {
        console.log('📊 Fetching fresh data from Golden Sheet CSV...');
        const response = await axios.get(GOLDEN_SHEET_CSV_URL, { 
            timeout: 15000,
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        });
        
        const csvData = response.data;
        
        // Debug: Log first 200 characters to see what we're receiving
        console.log('📝 First 200 chars of golden sheet response:', csvData.substring(0, 200));
        console.log('📝 Response type:', typeof csvData);
        
        // Parse CSV
        const records = parse(csvData, {
            columns: true,
            skip_empty_lines: true,
            trim: true,
            relax_quotes: true,
            relax_column_count: true
        });

        console.log('✅ Successfully parsed', records.length, 'rows from Golden Sheet');
        
        // Cache the data
        goldenSheetDataCache = {
            success: true,
            data: records,
            headers: Object.keys(records[0] || {})
        };
        lastGoldenCacheTime = Date.now();
        
        return goldenSheetDataCache;
        
    } catch (error) {
        console.error('❌ Error fetching from Golden Sheet CSV:', error.message);
        
        // If it's an axios error, log the response
        if (error.response) {
            console.error('📍 Status:', error.response.status);
            console.error('📍 Response preview:', error.response.data?.substring(0, 200));
        }
        
        // Return empty fallback data
        return {
            success: true,
            data: [],
            headers: [],
            fromCache: false
        };
    }
}

// API endpoint to get filtered subcategories
app.get('/api/filtered-subcategories', async (req, res) => {
    const { category, gender, ageGroup } = req.query;
    
    console.log('🔍 Filtering for:', { category, gender, ageGroup });
    
    try {
        const sheetResult = await getRealSheetData();
        const data = sheetResult.data;

        // Filter and get unique subcategories
        const filteredSubcategories = [];
        const seenSubcategories = new Set();

        data.forEach(row => {
            // Get category from the first unnamed column (empty string key)
            const rowCategory = row[''] || row.Category;
            const rowGender = row.Gender || row.gender;
            const rowAgeGroup = row['Age Group'] || row.AgeGroup || row.agegroup;
            const rowSubcategory = row.Subcategory || row['Sub Category'] || row.subcategory;

            if (rowCategory && rowCategory.toString().trim() === category &&
                rowGender && rowGender.toString().trim() === gender &&
                rowAgeGroup && rowAgeGroup.toString().trim() === ageGroup &&
                rowSubcategory) {
                
                const subcat = rowSubcategory.toString().trim();
                if (!seenSubcategories.has(subcat)) {
                    filteredSubcategories.push(subcat);
                    seenSubcategories.add(subcat);
                }
            }
        });

        console.log('📋 Found', filteredSubcategories.length, 'subcategories:', filteredSubcategories);

        res.json({
            success: true,
            subcategories: filteredSubcategories.sort()
        });
    } catch (error) {
        console.error('❌ Error fetching subcategories:', error.message);
        res.status(500).json({ 
            success: false, 
            error: 'Failed to fetch subcategories: ' + error.message 
        });
    }
});

// API endpoint to find matching URL
app.get('/api/find-url', async (req, res) => {
    const { category, gender, ageGroup, subcategory } = req.query;
    
    console.log('🔍 Finding URL for:', { category, gender, ageGroup, subcategory });
    
    try {
        const sheetResult = await getRealSheetData();
        const data = sheetResult.data;

        // Find matching row
        const matchingRow = data.find(row => {
            const rowCategory = row[''] || row.Category;
            const rowGender = row.Gender || row.gender;
            const rowAgeGroup = row['Age Group'] || row.AgeGroup || row.agegroup;
            const rowSubcategory = row.Subcategory || row['Sub Category'] || row.subcategory;
            const rowURL = row.URL || row.Url || row.url;

            return (rowCategory && rowCategory.toString().trim() === category &&
                    rowGender && rowGender.toString().trim() === gender &&
                    rowAgeGroup && rowAgeGroup.toString().trim() === ageGroup &&
                    rowSubcategory && rowSubcategory.toString().trim() === subcategory);
        });

        const url = matchingRow ? (matchingRow.URL || matchingRow.Url || matchingRow.url) : null;

        if (matchingRow && url) {
            res.json({
                success: true,
                url: url,
                found: true
            });
        } else {
            res.json({
                success: true,
                url: null,
                found: false,
                message: 'No matching URL found for the selected criteria'
            });
        }
    } catch (error) {
        console.error('❌ Error finding URL:', error.message);
        res.status(500).json({ 
            success: false, 
            error: 'Failed to find URL: ' + error.message 
        });
    }
});

// Helper function to clean and format scraped data for frontend
function formatScrapedData(rawData) {
    // Helper to clean keys - remove newlines, special characters, and extra spaces
    const cleanKey = (key) => {
        return key
            .replace(/\n/g, ' ')
            .replace(/\u200f|\u200e/g, '') // Remove RTL/LTR marks
            .replace(/\s+/g, ' ')
            .replace(/\s*:\s*$/, '')
            .trim();
    };

    // Helper to clean object keys
    const cleanObjectKeys = (obj) => {
        if (!obj || typeof obj !== 'object') return obj;
        
        const cleaned = {};
        for (let key in obj) {
            const cleanedKey = cleanKey(key);
            cleaned[cleanedKey] = obj[key];
        }
        return cleaned;
    };

    // Format product details as array for easier frontend rendering
    const formatDetailsAsArray = (obj) => {
        if (!obj || typeof obj !== 'object') return [];
        
        return Object.entries(obj)
            .filter(([key, value]) => key && value && key !== 'status')
            .map(([key, value]) => ({
                label: cleanKey(key),
                value: value
            }));
    };

    return {
        success: true,
        product: {
            title: rawData.title || 'N/A',
            description: rawData.description || 'N/A',
            asin: rawData.asin || 'N/A',
            images: rawData.images || []
        },
        details: {
            featureBullets: rawData.bullets || [],
            productDetails: formatDetailsAsArray(rawData.productDetails),
            manufacturingDetails: formatDetailsAsArray(rawData.manufacturingDetails),
            additionalInfo: formatDetailsAsArray(rawData.additionalInfo)
        },
        // Also include cleaned raw format for flexibility
        raw: {
            productDetails: cleanObjectKeys(rawData.productDetails),
            manufacturingDetails: cleanObjectKeys(rawData.manufacturingDetails),
            additionalInfo: cleanObjectKeys(rawData.additionalInfo)
        }
    };
}

// Helper function to format details as array for frontend
function formatDetailsAsArray(obj) {
    if (!obj || typeof obj !== 'object') return [];
    
    return Object.entries(obj)
        .filter(([key, value]) => key && value && key !== 'status')
        .map(([key, value]) => ({
            label: key.trim(),
            value: value
        }));
}

// Dual Scraper Integration - amz_scraper for images, scraper.py for other data
app.post('/api/scrape-product', async (req, res) => {
    let { url } = req.body;
    
    if (!url) {
        return res.status(400).json({ success: false, error: 'URL is required' });
    }

    // Fix URL if missing scheme
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        if (url.startsWith('www.') || url.startsWith('amazon.')) {
            url = 'https://' + url;
            console.log('🔧 Fixed URL scheme:', url);
        } else {
            // Assume it's an Amazon URL
            url = 'https://' + url;
            console.log('🔧 Added https:// to URL:', url);
        }
    }

    // Validate URL is an Amazon URL
    if (!url.includes('amazon.')) {
        return res.status(400).json({ 
            success: false, 
            error: 'Please provide a valid Amazon product URL',
            userMessage: 'The URL must be from Amazon.in or Amazon.com'
        });
    }

    console.log('🔍 Starting dual scraping for URL:', url);

    try {
        // Run both scrapers in parallel
        const scraperPromise = new Promise((resolve, reject) => {
            const scraperProcess = spawn('python3', ['scraper.py', url]);
            let scraperData = '';
            let scraperError = '';

            scraperProcess.stdout.on('data', (data) => {
                scraperData += data.toString();
            });

            scraperProcess.stderr.on('data', (data) => {
                scraperError += data.toString();
                console.log('scraper.py stderr:', data.toString());
            });

            scraperProcess.on('close', (code) => {
                if (code === 0 && scraperData) {
                    try {
                        resolve(JSON.parse(scraperData));
                    } catch (e) {
                        reject(new Error('Failed to parse scraper.py output'));
                    }
                } else {
                    reject(new Error(`scraper.py failed with code ${code}`));
                }
            });
        });

        const amzScraperPromise = new Promise((resolve, reject) => {
            const amzProcess = spawn('python3', ['amz_scraper.py', url, '--formatted']);
            let amzData = '';
            let amzError = '';

            amzProcess.stdout.on('data', (data) => {
                amzData += data.toString();
            });

            amzProcess.stderr.on('data', (data) => {
                amzError += data.toString();
                console.log('amz_scraper.py stderr:', data.toString());
            });

            amzProcess.on('close', (code) => {
                if (code === 0 && amzData) {
                    try {
                        resolve(JSON.parse(amzData));
                    } catch (e) {
                        reject(new Error('Failed to parse amz_scraper.py output'));
                    }
                } else {
                    reject(new Error(`amz_scraper.py failed with code ${code}`));
                }
            });
        });

        // Wait for both scrapers to complete - use allSettled to handle partial failures
        const [scraperResult, amzScraperResult] = await Promise.allSettled([scraperPromise, amzScraperPromise]);

        console.log('📊 Scraping Results Summary:');
        console.log('  scraper.py:', scraperResult.status === 'fulfilled' ? '✅ SUCCESS' : '❌ FAILED');
        console.log('  amz_scraper.py:', amzScraperResult.status === 'fulfilled' ? '✅ SUCCESS' : '❌ FAILED');

        // Extract data from fulfilled promises
        const scraperData = scraperResult.status === 'fulfilled' ? scraperResult.value : null;
        const amzData = amzScraperResult.status === 'fulfilled' ? amzScraperResult.value : null;

        // Log errors if any
        if (scraperResult.status === 'rejected') {
            console.error('  ❌ scraper.py error:', scraperResult.reason?.message || scraperResult.reason);
        }
        if (amzScraperResult.status === 'rejected') {
            console.error('  ❌ amz_scraper.py error:', amzScraperResult.reason?.message || amzScraperResult.reason);
        }

        // Check if we got ANY data at all 
        if (!scraperData && !amzData) {
            console.error('❌ Both scrapers failed completely');
            
            // Extract error messages from both scrapers
            const scraperError = scraperResult.status === 'rejected' 
                ? scraperResult.reason?.message || 'Unknown error'
                : null;
            const amzScraperError = amzScraperResult.status === 'rejected' 
                ? amzScraperResult.reason?.message || 'Unknown error'
                : null;
            
            // Return user-friendly error message
            return res.json({
                success: false,
                error: 'Please try again. Unable to load product information from Amazon.',
                userMessage: 'The product page could not be loaded. Please check the URL and try again in a few moments.',
                technicalDetails: {
                    scraperError: scraperError,
                    amzScraperError: amzScraperError,
                    timestamp: new Date().toISOString()
                }
            });
        }

        // Enhanced merge with multiple fallbacks
        const getAboutThisItem = (scraperData, amzData) => {
            // Priority 1: scraper.py bullets
            if (scraperData?.bullets && scraperData.bullets.length > 0) {
                return scraperData.bullets;
            }
            
            // Priority 2: amz_scraper aboutThisItem
            if (amzData?.product?.aboutThisItem && amzData.product.aboutThisItem.length > 0) {
                return amzData.product.aboutThisItem;
            }
            
            // Priority 3: Check raw data for any bullet-like content
            if (scraperData?.raw && scraperData.raw.bullets && scraperData.raw.bullets.length > 0) {
                return scraperData.raw.bullets;
            }
            
            // Final fallback
            return ['No "About this item" information available.'];
        };

        // Extract images with fallback - prefer amz_scraper, fallback to scraper.py 
        const images = (amzData?.product?.images?.length > 0 
            ? amzData.product.images 
            : scraperData?.images || []);

        // Merge results with fallback logic (combined approach)
        const mergedResult = {
            success: true,
            warnings: [],
            product: {
                title: scraperData?.title || amzData?.product?.title || 'N/A',
                description: scraperData?.description || amzData?.product?.description || 'N/A',
                images: images,
                aboutThisItem: getAboutThisItem(scraperData, amzData),
                asin: scraperData?.asin || 'N/A'
            },
            details: {
                featureBullets: getAboutThisItem(scraperData, amzData),
                productDetails: formatDetailsAsArray(scraperData?.productDetails),
                manufacturingDetails: formatDetailsAsArray(scraperData?.manufacturingDetails),
                additionalInfo: formatDetailsAsArray(scraperData?.additionalInfo)
            },
            raw: {
                productDetails: scraperData?.productDetails || {},
                manufacturingDetails: scraperData?.manufacturingDetails || {},
                additionalInfo: scraperData?.additionalInfo || {},
                bullets: scraperData?.bullets || [],
                amzScraperBullets: amzData?.product?.aboutThisItem || []
            }
        };

        // Add warnings if scrapers failed 
        if (scraperResult.status === 'rejected') {
            mergedResult.warnings.push('Text scraper failed - data may be incomplete');
        }
        if (amzScraperResult.status === 'rejected') {
            mergedResult.warnings.push('Image scraper failed - using fallback images');
        }

        // Enhanced logging to debug the aboutThisItem data
        console.log('📋 About this item bullets:', {
            hasBullets: !!scraperData?.bullets,
            bulletCount: scraperData?.bullets ? scraperData.bullets.length : 0,
            bulletsSample: scraperData?.bullets ? scraperData.bullets.slice(0, 2) : 'none'
        });

        // Log summary
        console.log('✅ Scraping completed with merged data:');
        console.log('  Title:', mergedResult.product.title !== 'N/A' ? '✅' : '❌');
        console.log('  Images:', images.length, 'found');
        console.log('  Product Details:', Object.keys(scraperData?.productDetails || {}).length, 'fields');
        console.log('  Warnings:', mergedResult.warnings.length);
        
        res.json(mergedResult);

    } catch (error) {
        console.error('❌ Scraping error:', error.message);
        res.json({
            success: false,
            error: 'Please try again. Unable to process the product page.',
            userMessage: 'An unexpected error occurred while loading the product. Please try again.',
            technicalDetails: {
                message: error.message,
                timestamp: new Date().toISOString()
            }
        });
    }
});

// API endpoint to get all sheet data
app.get('/api/sheet-data', async (req, res) => {
    try {
        const sheetResult = await getRealSheetData();
        
        res.json({
            success: true,
            data: sheetResult.data,
            headers: sheetResult.headers,
            totalRows: sheetResult.data.length,
            fromCache: sheetResult.fromCache || false
        });
    } catch (error) {
        console.error('❌ Error fetching sheet data:', error.message);
        res.status(500).json({ 
            success: false, 
            error: 'Failed to fetch sheet data: ' + error.message 
        });
    }
});

// API endpoint to get golden sheet data (Page 2)
app.get('/api/golden-sheet-data', async (req, res) => {
    try {
        const sheetResult = await getGoldenSheetData();
        
        res.json({
            success: true,
            data: sheetResult.data,
            headers: sheetResult.headers,
            totalRows: sheetResult.data.length,
            fromCache: sheetResult.fromCache || false
        });
    } catch (error) {
        console.error('❌ Error fetching golden sheet data:', error.message);
        res.status(500).json({ 
            success: false, 
            error: 'Failed to fetch golden sheet data: ' + error.message 
        });
    }
});

// Debug endpoint
app.get('/api/debug-sheet', async (req, res) => {
    try {
        const sheetResult = await getRealSheetData();
        
        const allData = sheetResult.data.map(row => ({
            firstColumn: row[''], // Category column
            gender: row.Gender,
            ageGroup: row['Age Group'],
            subcategory: row.Subcategory,
            url: row.URL,
            raw: row
        }));

        const uniqueValues = {
            categories: [...new Set(allData.map(row => row.firstColumn).filter(Boolean))],
            genders: [...new Set(allData.map(row => row.gender).filter(Boolean))],
            ageGroups: [...new Set(allData.map(row => row.ageGroup).filter(Boolean))],
            subcategories: [...new Set(allData.map(row => row.subcategory).filter(Boolean))]
        };

        res.json({
            success: true,
            totalRows: sheetResult.data.length,
            uniqueValues: uniqueValues,
            sampleData: allData.slice(0, 5),
            rawFirstRow: sheetResult.data[0]
        });
    } catch (error) {
        res.status(500).json({ 
            success: false, 
            error: 'Failed to debug sheet: ' + error.message 
        });
    }
});

app.post('/api/submit-listing', (req, res) => {
    console.log('📦 Listing submitted:', req.body);
    res.json({ 
        success: true, 
        message: 'Listing submitted successfully',
        submittedData: req.body
    });
});

// Google Sheets Upload API - Append CSV data to a common sheet
app.post('/api/upload-to-sheets', async (req, res) => {
    const { csvData, sheetName } = req.body;
    
    if (!csvData || !Array.isArray(csvData)) {
        return res.status(400).json({ 
            success: false, 
            error: 'csvData (array of rows) is required' 
        });
    }

    const SPREADSHEET_ID = '1-azsWJHPgwUicaX1bdJUoH-4bFxR-sFmwtmFeeDUXgc';
    const COMMON_SHEET_NAME = 'Product_Data'; // Common sheet name for all products

    try {
        console.log('📊 Uploading to Google Sheets...');
        console.log('📋 Sheet name:', COMMON_SHEET_NAME);
        console.log('📦 Data rows:', csvData.length);

        // Load service account credentials
        const auth = new google.auth.GoogleAuth({
            keyFile: path.join(__dirname, 'credentials.json'),
            scopes: ['https://www.googleapis.com/auth/spreadsheets'],
        });

        const sheets = google.sheets({ version: 'v4', auth });

        // Check if the common sheet exists
        let sheetExists = false;
        let sheetId = null;
        
        try {
            const response = await sheets.spreadsheets.get({
                spreadsheetId: SPREADSHEET_ID,
            });
            
            const sheet = response.data.sheets.find(
                (s) => s.properties.title === COMMON_SHEET_NAME
            );
            
            if (sheet) {
                sheetExists = true;
                sheetId = sheet.properties.sheetId;
                console.log('✅ Sheet already exists, will append data');
            }
        } catch (error) {
            console.log('📄 Sheet does not exist, will create new one');
        }

        // Step 1: Create the sheet if it doesn't exist
        if (!sheetExists) {
            console.log('📄 Creating new sheet:', COMMON_SHEET_NAME);
            const createResponse = await sheets.spreadsheets.batchUpdate({
                spreadsheetId: SPREADSHEET_ID,
                requestBody: {
                    requests: [
                        {
                            addSheet: {
                                properties: {
                                    title: COMMON_SHEET_NAME,
                                },
                            },
                        },
                    ],
                },
            });

            sheetId = createResponse.data.replies[0].addSheet.properties.sheetId;
            console.log('✅ New sheet created:', COMMON_SHEET_NAME);

            // Write headers for new sheet
            console.log('📝 Writing headers to new sheet...');
            await sheets.spreadsheets.values.update({
                spreadsheetId: SPREADSHEET_ID,
                range: `${COMMON_SHEET_NAME}!A1`,
                valueInputOption: 'RAW',
                requestBody: {
                    values: [csvData[0]], // First row is headers
                },
            });

            // Format the header row (make it bold and freeze)
            await sheets.spreadsheets.batchUpdate({
                spreadsheetId: SPREADSHEET_ID,
                requestBody: {
                    requests: [
                        {
                            repeatCell: {
                                range: {
                                    sheetId: sheetId,
                                    startRowIndex: 0,
                                    endRowIndex: 1,
                                },
                                cell: {
                                    userEnteredFormat: {
                                        textFormat: {
                                            bold: true,
                                        },
                                        backgroundColor: {
                                            red: 0.9,
                                            green: 0.9,
                                            blue: 0.9,
                                        },
                                    },
                                },
                                fields: 'userEnteredFormat(textFormat,backgroundColor)',
                            },
                        },
                        {
                            updateSheetProperties: {
                                properties: {
                                    sheetId: sheetId,
                                    gridProperties: {
                                        frozenRowCount: 1,
                                    },
                                },
                                fields: 'gridProperties.frozenRowCount',
                            },
                        },
                    ],
                },
            });
            console.log('✅ Headers formatted and frozen');
        }

        // Step 2: Get the current number of rows in the sheet
        const existingDataResponse = await sheets.spreadsheets.values.get({
            spreadsheetId: SPREADSHEET_ID,
            range: `${COMMON_SHEET_NAME}!A:A`,
        });

        const existingRowCount = existingDataResponse.data.values ? existingDataResponse.data.values.length : 0;
        const nextRow = existingRowCount + 1;

        console.log('📊 Current rows in sheet:', existingRowCount);
        console.log('📝 Will append starting at row:', nextRow);

        // Step 3: Append the data (skip headers if sheet already exists)
        const dataToAppend = sheetExists ? csvData.slice(1) : csvData.slice(1);
        
        if (dataToAppend.length > 0) {
            await sheets.spreadsheets.values.append({
                spreadsheetId: SPREADSHEET_ID,
                range: `${COMMON_SHEET_NAME}!A${nextRow}`,
                valueInputOption: 'RAW',
                insertDataOption: 'INSERT_ROWS',
                requestBody: {
                    values: dataToAppend,
                },
            });

            console.log('✅ Data appended successfully');
        }

        const sheetUrl = `https://docs.google.com/spreadsheets/d/${SPREADSHEET_ID}/edit#gid=${sheetId}`;

        res.json({
            success: true,
            message: sheetExists ? 'Product data appended to existing sheet' : 'New sheet created and data added',
            sheetUrl: sheetUrl,
            tabName: COMMON_SHEET_NAME,
            rowCount: dataToAppend.length,
            totalRowsNow: existingRowCount + dataToAppend.length,
        });

    } catch (error) {
        console.error('❌ Error uploading to Google Sheets:', error.message);
        console.error('Stack:', error.stack);
        res.status(500).json({
            success: false,
            error: 'Failed to upload to Google Sheets: ' + error.message,
        });
    }
});

// Helper function to get sheet ID by name
async function getSheetId(sheets, spreadsheetId, sheetName) {
    try {
        const response = await sheets.spreadsheets.get({
            spreadsheetId: spreadsheetId,
        });
        
        const sheet = response.data.sheets.find(
            (s) => s.properties.title === sheetName
        );
        
        return sheet ? sheet.properties.sheetId : null;
    } catch (error) {
        console.error('Error getting sheet ID:', error.message);
        return null;
    }
}

// Serve main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Serve scraper page
app.get('/scraper', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'scraper.html'));
});

// Serve test page
app.get('/test-sheet', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'simple-test.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`📊 Google Sheets integration: ACTIVE`);
    console.log(`🔍 Python Scraper: READY (with mock data fallback)`);
    console.log(`💡 Main App: http://localhost:${PORT}`);
    console.log(`🔧 Scraper UI: http://localhost:${PORT}/scraper`);
    console.log(`🐛 Debug: http://localhost:${PORT}/api/debug-sheet`);
});