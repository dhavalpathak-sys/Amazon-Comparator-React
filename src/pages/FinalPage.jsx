

import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import html2pdf from 'html2pdf.js';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const FinalPage = () => {
  const [comparisonData, setComparisonData] = useState(null);
  const [userInfo, setUserInfo] = useState(null);
  const [activeTab, setActiveTab] = useState('original');
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [allImages, setAllImages] = useState([]);
  const [isExporting, setIsExporting] = useState(false);
  const contentRef = useRef(null);
  const navigate = useNavigate();

  const handleDownload = () => {
    const element = contentRef.current;
    const opt = {
      margin: 1,
      filename: 'product-comparison-report.pdf',
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(element).save();
  };

  const handleProceedAndExportToSheets = async () => {
    if (!comparisonData) {
      alert('No data available to export');
      return;
    }

    setIsExporting(true);

    // Helper function to safely extract value
    const safeValue = (value) => {
      if (value === null || value === undefined) return 'N/A';
      return String(value);
    };

    // Helper to find value in productDetails or manufacturingDetails
    const findDetailValue = (labelSearch) => {
      if (comparisonData.productDetails) {
        const detail = comparisonData.productDetails.find(d => 
          d.label.toLowerCase().includes(labelSearch.toLowerCase())
        );
        if (detail) return detail.value;
      }
      if (comparisonData.manufacturingDetails) {
        const detail = comparisonData.manufacturingDetails.find(d => 
          d.label.toLowerCase().includes(labelSearch.toLowerCase())
        );
        if (detail) return detail.value;
      }
      return 'N/A';
    };

    // Extract features as bullet points
    const getBulletPoint = (index) => {
      if (comparisonData.features && comparisonData.features[index]) {
        return comparisonData.features[index];
      }
      return 'N/A';
    };

    // Extract special features
    const getSpecialFeature = (index) => {
      if (comparisonData.features && comparisonData.features.length > 5) {
        const featureIndex = 5 + index;
        if (comparisonData.features[featureIndex]) {
          return comparisonData.features[featureIndex];
        }
      }
      return 'N/A';
    };

    // Get image URLs
    const getImageURL = (index) => {
      if (allImages && allImages[index]) {
        return allImages[index].url;
      }
      return 'N/A';
    };

    // Get current date in day, month, year format
    const currentDate = new Date();
    const day = currentDate.getDate();
    const month = currentDate.getMonth() + 1; // Months are 0-indexed
    const year = currentDate.getFullYear();
    const formattedDate = `${day}/${month}/${year}`;

    // Map frontend data to Amazon template columns
    const columnHeaders = [
      'Date', 'Number_of_attributes_with_errors', 'Number_of_attributes_with_other_suggestions', 'Product_Sub_type',
      'Seller_SKU', 'Brand_Name', 'Update_Delete', 'Manufacturer_Part_Number', 'Product_ID', 'Product_ID_Type',
      'Item_Name', 'Product_Description', 'Manufacturer', 'Recommended_Browse_Nodes', 'Closure_Type',
      'Model_Name', 'Model_Number', 'Product_Care_Instructions', 'Your_price', 'Quantity',
      'Apparel_Size_Body_Type', 'Target_Gender', 'Age_Range_Description', 'Apparel_Size_System',
      'Apparel_Size_Class', 'Apparel_Size_Value', 'Apparel_Size_To_Range', 'Main_Image_URL',
      'Other_Image_URL1', 'Other_Image_URL2', 'Other_Image_URL3', 'Other_Image_URL4', 'Other_Image_URL5',
      'Other_Image_URL6', 'Other_Image_URL7', 'Other_Image_URL8', 'Swatch_Image_URL', 'Parentage',
      'Parent_SKU', 'Relationship_Type', 'Variation_Theme', 'Bullet_Point1', 'Bullet_Point2',
      'Bullet_Point3', 'Bullet_Point4', 'Bullet_Point5', 'Generic_Keyword', 'Style',
      'Packer_Contact_Information', 'Special_Features1', 'Special_Features2', 'Special_Features3',
      'Special_Features4', 'Special_Features5', 'Design', 'Item_Type_Name', 'Occasion1', 'Occasion2',
      'Occasion3', 'Occasion4', 'Occasion5', 'Duration_Unit', 'Neck_Style', 'Colour_Map', 'Season',
      'Team_Name', 'Importer_Contact_Information', 'Material_Type', 'Duration', 'Occasion_Lifestyle',
      'Number_Of_Pockets', 'Manufacturer_Contact', 'Fur_Description', 'Fitting_type', 'Collar_Style',
      'Color', 'Embellishment_Feature1', 'Embellishment_Feature2', 'Embellishment_Feature3',
      'Embellishment_Feature4', 'Embellishment_Feature5', 'Weave_Type', 'Back_Style', 'Theme', 'Gender',
      'League_Name', 'Special_Sizes', 'Embroidery_Type', 'Included_Components1', 'Included_Components2',
      'Included_Components3', 'Included_Components4', 'Included_Components5', 'Sleeve_Type', 'Size',
      'Product_Lifecycle_Supply_Type', 'Pattern', 'Maximum_Seller_Allowed_Price',
      'flash_point_unit_of_measure', 'Is_exempt_from_supplier_declared_external_product_identifier',
      'Skip_Offer', 'Minimum_Advertised_Price_Display', 'Minimum_Seller_Allowed_Price', 'Item_Weight',
      'Item_Weight_Unit', 'Item_Width_Unit_Of_Measure', 'Item_Width', 'Item_Height', 'Generic_Size',
      'Chest_Size_Unit', 'Chest_Size', 'Unit_Count', 'Item_Length_Description',
      'Item_Height_Unit_Of_Measure', 'Item_Length_Unit_Of_Measure', 'Item_Length', 'Unit_Count_Type',
      'Item_Length_Longer_Edge', 'Item_Length_Unit', 'Fulfillment_Centre_ID', 'Package_Weight',
      'Package_Weight_Unit_Of_Measure', 'Package_Height', 'Item_Package_height_Unit_of_Measure',
      'Package_Length', 'Package_Length_Unit_Of_Measure', 'package_width',
      'Item_Package_width_Unit_of_Measure', 'Country_of_Origin', 'Are_batteries_required',
      'Dangerous_Goods_Regulations1', 'Dangerous_Goods_Regulations2', 'Dangerous_Goods_Regulations3',
      'Dangerous_Goods_Regulations4', 'Dangerous_Goods_Regulations5', 'Hazmat', 'Safety_Data_Sheet_URL',
      'Compliance_Media_Content_Type1', 'Compliance_Media_Content_Type2', 'Compliance_Media_Content_Type3',
      'Compliance_Media_Content_Type4', 'Compliance_Media_Content_Type5', 'Compliance_Media_Content_Type6',
      'Compliance_Media_Content_Type7', 'Compliance_Media_Content_Type8', 'Compliance_Media_Content_Type9',
      'Compliance_Media_Content_Type10', 'Compliance_Media_Content_Type11', 'Compliance_Media_Content_Type12',
      'Compliance_Media_Content_Type13', 'Compliance_Media_Content_Type14', 'Compliance_Media_Content_Type15',
      'Compliance_Media_Content_Type16', 'Compliance_Media_Content_Type17', 'Fabric_Type', 'Flash_point_C',
      'Specific_Uses_for_Product', 'External_Product_Information', 'Material_Fabric_Regulations1',
      'Material_Fabric_Regulations2', 'Material_Fabric_Regulations3', 'Compliance_Media_Source_Location1',
      'Compliance_Media_Source_Location2', 'Compliance_Media_Source_Location3',
      'Compliance_Media_Source_Location4', 'Compliance_Media_Source_Location5',
      'Compliance_Media_Source_Location6', 'Compliance_Media_Source_Location7',
      'Compliance_Media_Source_Location8', 'Compliance_Media_Source_Location9',
      'Compliance_Media_Source_Location10', 'Compliance_Media_Source_Location11',
      'Compliance_Media_Source_Location12', 'Compliance_Media_Source_Location13',
      'Compliance_Media_Source_Location14', 'Compliance_Media_Source_Location15',
      'Compliance_Media_Source_Location16', 'Compliance_Media_Source_Location17',
      'Compliance_Media_Language1', 'Compliance_Media_Language2', 'Compliance_Media_Language3',
      'Compliance_Media_Language4', 'Compliance_Media_Language5', 'Compliance_Media_Language6',
      'Compliance_Media_Language7', 'Compliance_Media_Language8', 'Compliance_Media_Language9',
      'Compliance_Media_Language10', 'Compliance_Media_Language11', 'Compliance_Media_Language12',
      'Compliance_Media_Language13', 'Compliance_Media_Language14', 'Compliance_Media_Language15',
      'Compliance_Media_Language16', 'Compliance_Media_Language17', 'GHS_Class', 'Condition',
      'Condition_Note', 'Handling_Time', 'Release_Date', 'Restock_Date', 'Product_Tax_Code',
      'Sale_Price', 'Sale_Start_Date', 'Sale_End_Date', 'Max_Order_Quantity', 'Number_of_Items',
      'Can_Be_Gift_Messaged', 'Is_Gift_Wrap_Available', 'Minimum_Advertised_Price', 'List_Price',
      'Currency', 'Offer_End_Date', 'Shipping_Template', 'Offer_Start_Date', 'Maximum_Retail_Price',
      'Business_Price', 'Quantity_Price_Type', 'Quantity_Price_1', 'Quantity_Lower_Bound_1',
      'Quantity_Price_2', 'Quantity_Lower_Bound_2', 'Quantity_Price_3', 'Quantity_Lower_Bound_3',
      'Quantity_Price_4', 'Quantity_Lower_Bound_4', 'Quantity_Price_5', 'Quantity_Lower_Bound_5',
      'Pricing_Action', 'United_Nations_Standard_Products_and_Services_Code', 'National_Stock_Number'
    ];

    // Map data values to columns
    const dataRow = [
      formattedDate, // Date
      'N/A', // Number_of_attributes_with_errors
      'N/A', // Number_of_attributes_with_other_suggestions
      safeValue(comparisonData.subcategory), // Product_Sub_type
      safeValue(comparisonData.originalAsin), // Seller_SKU
      safeValue(comparisonData.brandComparison?.original), // Brand_Name
      'Update', // Update_Delete
      findDetailValue('part number'), // Manufacturer_Part_Number
      safeValue(comparisonData.originalAsin), // Product_ID
      'ASIN', // Product_ID_Type
      safeValue(comparisonData.originalTitle), // Item_Name
      safeValue(comparisonData.description || comparisonData.notes), // Product_Description
      findDetailValue('manufacturer'), // Manufacturer
      'N/A', // Recommended_Browse_Nodes
      findDetailValue('closure'), // Closure_Type
      findDetailValue('model name'), // Model_Name
      findDetailValue('model number'), // Model_Number
      findDetailValue('care instructions'), // Product_Care_Instructions
      safeValue(comparisonData.originalPrice), // Your_price
      '1', // Quantity
      'N/A', // Apparel_Size_Body_Type
      safeValue(findDetailValue('gender') || comparisonData.department), // Target_Gender
      findDetailValue('age range'), // Age_Range_Description
      'N/A', // Apparel_Size_System
      'N/A', // Apparel_Size_Class
      findDetailValue('size'), // Apparel_Size_Value
      'N/A', // Apparel_Size_To_Range
      getImageURL(0), // Main_Image_URL
      getImageURL(1), // Other_Image_URL1
      getImageURL(2), // Other_Image_URL2
      getImageURL(3), // Other_Image_URL3
      getImageURL(4), // Other_Image_URL4
      getImageURL(5), // Other_Image_URL5
      getImageURL(6), // Other_Image_URL6
      getImageURL(7), // Other_Image_URL7
      getImageURL(8), // Other_Image_URL8
      'N/A', // Swatch_Image_URL
      'N/A', // Parentage
      'N/A', // Parent_SKU
      'N/A', // Relationship_Type
      'N/A', // Variation_Theme
      getBulletPoint(0), // Bullet_Point1
      getBulletPoint(1), // Bullet_Point2
      getBulletPoint(2), // Bullet_Point3
      getBulletPoint(3), // Bullet_Point4
      getBulletPoint(4), // Bullet_Point5
      safeValue(comparisonData.category), // Generic_Keyword
      findDetailValue('style'), // Style
      findDetailValue('packer'), // Packer_Contact_Information
      getSpecialFeature(0), // Special_Features1
      getSpecialFeature(1), // Special_Features2
      getSpecialFeature(2), // Special_Features3
      getSpecialFeature(3), // Special_Features4
      getSpecialFeature(4), // Special_Features5
      findDetailValue('design'), // Design
      safeValue(comparisonData.department), // Item_Type_Name
      findDetailValue('occasion'), // Occasion1
      'N/A', // Occasion2
      'N/A', // Occasion3
      'N/A', // Occasion4
      'N/A', // Occasion5
      'N/A', // Duration_Unit
      findDetailValue('neck'), // Neck_Style
      findDetailValue('color') || findDetailValue('colour'), // Colour_Map
      findDetailValue('season'), // Season
      'N/A', // Team_Name
      findDetailValue('importer'), // Importer_Contact_Information
      findDetailValue('material'), // Material_Type
      'N/A', // Duration
      'N/A', // Occasion_Lifestyle
      findDetailValue('pockets'), // Number_Of_Pockets
      findDetailValue('manufacturer'), // Manufacturer_Contact
      findDetailValue('fur'), // Fur_Description
      findDetailValue('fit'), // Fitting_type
      findDetailValue('collar'), // Collar_Style
      findDetailValue('color') || findDetailValue('colour'), // Color
      'N/A', // Embellishment_Feature1
      'N/A', // Embellishment_Feature2
      'N/A', // Embellishment_Feature3
      'N/A', // Embellishment_Feature4
      'N/A', // Embellishment_Feature5
      findDetailValue('weave'), // Weave_Type
      'N/A', // Back_Style
      findDetailValue('theme'), // Theme
      safeValue(findDetailValue('gender') || comparisonData.department), // Gender
      'N/A', // League_Name
      'N/A', // Special_Sizes
      'N/A', // Embroidery_Type
      'N/A', // Included_Components1
      'N/A', // Included_Components2
      'N/A', // Included_Components3
      'N/A', // Included_Components4
      'N/A', // Included_Components5
      findDetailValue('sleeve'), // Sleeve_Type
      findDetailValue('size'), // Size
      'N/A', // Product_Lifecycle_Supply_Type
      findDetailValue('pattern'), // Pattern
      'N/A', // Maximum_Seller_Allowed_Price
      'N/A', // flash_point_unit_of_measure
      'N/A', // Is_exempt_from_supplier_declared_external_product_identifier
      'N/A', // Skip_Offer
      'N/A', // Minimum_Advertised_Price_Display
      'N/A', // Minimum_Seller_Allowed_Price
      findDetailValue('weight'), // Item_Weight
      'N/A', // Item_Weight_Unit
      'N/A', // Item_Width_Unit_Of_Measure
      findDetailValue('width'), // Item_Width
      findDetailValue('height'), // Item_Height
      findDetailValue('size'), // Generic_Size
      'N/A', // Chest_Size_Unit
      findDetailValue('chest'), // Chest_Size
      'N/A', // Unit_Count
      findDetailValue('length'), // Item_Length_Description
      'N/A', // Item_Height_Unit_Of_Measure
      'N/A', // Item_Length_Unit_Of_Measure
      findDetailValue('length'), // Item_Length
      'N/A', // Unit_Count_Type
      'N/A', // Item_Length_Longer_Edge
      'N/A', // Item_Length_Unit
      'N/A', // Fulfillment_Centre_ID
      findDetailValue('package weight'), // Package_Weight
      'N/A', // Package_Weight_Unit_Of_Measure
      findDetailValue('package height'), // Package_Height
      'N/A', // Item_Package_height_Unit_of_Measure
      findDetailValue('package length'), // Package_Length
      'N/A', // Package_Length_Unit_Of_Measure
      findDetailValue('package width'), // package_width
      'N/A', // Item_Package_width_Unit_of_Measure
      findDetailValue('country') || findDetailValue('origin'), // Country_of_Origin
      'N/A', // Are_batteries_required
      'N/A', // Dangerous_Goods_Regulations1
      'N/A', // Dangerous_Goods_Regulations2
      'N/A', // Dangerous_Goods_Regulations3
      'N/A', // Dangerous_Goods_Regulations4
      'N/A', // Dangerous_Goods_Regulations5
      'N/A', // Hazmat
      'N/A', // Safety_Data_Sheet_URL
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', // Compliance_Media_Content_Type1-10
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', // Compliance_Media_Content_Type11-17
      findDetailValue('fabric'), // Fabric_Type
      'N/A', // Flash_point_C
      'N/A', // Specific_Uses_for_Product
      'N/A', // External_Product_Information
      'N/A', 'N/A', 'N/A', // Material_Fabric_Regulations1-3
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', // Compliance_Media_Source_Location1-10
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', // Compliance_Media_Source_Location11-17
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', // Compliance_Media_Language1-10
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', // Compliance_Media_Language11-17
      'N/A', // GHS_Class
      'New', // Condition
      'N/A', // Condition_Note
      '2', // Handling_Time
      'N/A', // Release_Date
      'N/A', // Restock_Date
      'N/A', // Product_Tax_Code
      'N/A', // Sale_Price
      'N/A', // Sale_Start_Date
      'N/A', // Sale_End_Date
      'N/A', // Max_Order_Quantity
      '1', // Number_of_Items
      'N/A', // Can_Be_Gift_Messaged
      'N/A', // Is_Gift_Wrap_Available
      'N/A', // Minimum_Advertised_Price
      safeValue(comparisonData.originalPrice), // List_Price
      'INR', // Currency
      'N/A', // Offer_End_Date
      'N/A', // Shipping_Template
      'N/A', // Offer_Start_Date
      safeValue(comparisonData.originalPrice), // Maximum_Retail_Price
      'N/A', // Business_Price
      'N/A', // Quantity_Price_Type
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', // Quantity_Price/Bound 1-5
      'N/A', // Pricing_Action
      'N/A', // United_Nations_Standard_Products_and_Services_Code
      'N/A'  // National_Stock_Number
    ];

    // Upload to Google Sheets
    try {
      console.log('📤 Uploading to Google Sheets...');
      
      // Prepare data for Google Sheets API (array of arrays)
      const sheetData = [
        columnHeaders, // First row: headers
        dataRow        // Second row: data
      ];
      
      const response = await fetch('http://localhost:3000/api/upload-to-sheets', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          csvData: sheetData,
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        console.log('✅ Data uploaded to Google Sheets:', result.sheetUrl);
        console.log('📊 Total rows now:', result.totalRowsNow);
        
        // Open Google Sheets in a new tab
        if (result.sheetUrl) {
          window.open(result.sheetUrl, '_blank');
        }
        
        const message = result.totalRowsNow > 1 
          ? `Success! \n\n✅ Product data appended to Google Sheet\n\nSheet: ${result.tabName}\nTotal products: ${result.totalRowsNow - 1}\n\nOpening Google Sheet in new tab...`
          : `Success! \n\n✅ New sheet created and data uploaded\n\nSheet: ${result.tabName}\n\nOpening Google Sheet in new tab...`;
        
        // alert(message);
      } else {
        console.error('❌ Failed to upload to Google Sheets:', result.error);
        alert('Failed to upload to Google Sheets: ' + result.error);
      }
    } catch (error) {
      console.error('❌ Error uploading to Google Sheets:', error);
      alert('Failed to upload to Google Sheets. Please check console for details.');
    } finally {
      setIsExporting(false);
    }
  };

  const renderStars = (rating) => {
    return "⭐".repeat(rating) + "☆".repeat(5 - rating);
  };

  // Image slider navigation
  const nextImage = () => {
    if (allImages.length > 0) {
      setCurrentImageIndex((prev) => (prev + 1) % allImages.length);
    }
  };

  const prevImage = () => {
    if (allImages.length > 0) {
      setCurrentImageIndex((prev) => (prev - 1 + allImages.length) % allImages.length);
    }
  };

  const goToImage = (index) => {
    setCurrentImageIndex(index);
  };

  useEffect(() => {
    // Get user info from localStorage or set default
    const user = JSON.parse(localStorage.getItem('user')) || {
      name: "John Doe",
      email: "john.doe@example.com"
    };
    setUserInfo(user);

    // Load existing product from localStorage
    const storedExisting = localStorage.getItem('existingProduct');
    const existingProductData = storedExisting ? JSON.parse(storedExisting) : null;
    
    // Load all product images (existing + generated) from localStorage
    const storedAllImages = localStorage.getItem('allProductImages');
    let productImages = [];
    
    if (storedAllImages) {
      try {
        productImages = JSON.parse(storedAllImages);
        console.log('📸 Loaded all product images:', productImages);
      } catch (error) {
        console.error('Error parsing all product images:', error);
      }
    }
    
    // Fallback: If no combined images, try to build from individual sources
    if (productImages.length === 0) {
      // Try loading generated images
      const storedGeneratedImages = localStorage.getItem('generatedImages');
      const generatedImages = storedGeneratedImages ? JSON.parse(storedGeneratedImages) : [];
      
      // Add existing product images
      if (existingProductData?.images && Array.isArray(existingProductData.images)) {
        existingProductData.images.forEach((url, index) => {
          productImages.push({
            url: url,
            type: 'existing',
            label: `Existing Image ${index + 1}`
          });
        });
      }
      
      // Add generated images
      if (generatedImages.length > 0) {
        generatedImages.forEach((url, index) => {
          productImages.push({
            url: url,
            type: 'generated',
            label: `AI Generated ${index + 1}`
          });
        });
      }
      
      console.log('📦 Built combined images from individual sources:', productImages);
    }
    
    setAllImages(productImages);
    
    // Load generated images from localStorage (for backward compatibility)
    const storedGeneratedImages = localStorage.getItem('generatedImages');
    const generatedImages = storedGeneratedImages ? JSON.parse(storedGeneratedImages) : [];
    
    // Load gold standard product from localStorage
    const storedGoldStandard = localStorage.getItem('goldStandardProduct');
    const goldStandardData = storedGoldStandard ? JSON.parse(storedGoldStandard) : null;
    
    // Load product attributes
    const storedFilters = localStorage.getItem('productFilters');
    const productAttributes = storedFilters ? JSON.parse(storedFilters) : {};

    console.log('📦 Loaded existing product:', existingProductData);
    console.log('🎨 Loaded generated images:', generatedImages);
    console.log('⭐ Loaded gold standard:', goldStandardData);
    console.log('🏷️ Loaded attributes:', productAttributes);

    // Build comparison data from existing product
    if (existingProductData) {
      // Extract features from existing product ONLY (not gold standard)
      const existingFeatures = extractFeatures(existingProductData);
      console.log('✨ Features extracted from EXISTING product:', existingFeatures);
      console.log('🔍 Existing product features property:', existingProductData.features);
      
      const data = {
        // Original Product (Existing Product)
        originalTitle: existingProductData.title || 'N/A',
        originalPrice: extractPrice(existingProductData) || 0,
        originalAsin: existingProductData.asin || existingProductData.id || 'N/A',
        originalImage: (existingProductData.images && existingProductData.images[0]) || 'https://placehold.co/400x400/E5E7EB/6B7280?text=No+Image',
        originalRating: extractRating(existingProductData) || 0,
        originalReviews: extractReviews(existingProductData) || 0,
        
        // Competitor Product (Gold Standard - if available)
        competitorTitle: goldStandardData?.title || 'Gold Standard Product',
        competitorPrice: extractPrice(goldStandardData) || 0,
        competitorAsin: goldStandardData?.asin || goldStandardData?.id || 'N/A',
        competitorImage: (goldStandardData?.images && goldStandardData.images[0]) || 'https://placehold.co/400x400/E5E7EB/6B7280?text=No+Image',
        competitorRating: extractRating(goldStandardData) || 0,
        competitorReviews: extractReviews(goldStandardData) || 0,

        // Generated Images
        generatedImages: generatedImages || [],

        // Product Details
        description: existingProductData.description || 'No description available',
        features: existingFeatures,
        
        // Brand Information
        brandComparison: {
          original: existingProductData.brand || extractBrand(existingProductData) || 'N/A',
          competitor: goldStandardData?.brand || extractBrand(goldStandardData) || 'N/A'
        },
        
        // Product Attributes
        productAttributes: productAttributes,
        
        // Additional Details
        productDetails: existingProductData.productDetailsArray || [],
        manufacturingDetails: existingProductData.manufacturingDetailsArray || [],
        
        // Metadata
        department: extractDepartment(existingProductData) || productAttributes.Category || 'General',
        category: productAttributes.Category || 'N/A',
        subcategory: productAttributes.Subcategory || productAttributes.SubCategory || 'N/A',
        
        // Sizes and Colors (extract from product details or use defaults)
        sizes: extractSizes(existingProductData) || ['S', 'M', 'L', 'XL', 'XXL'],
        colors: extractColors(existingProductData) || ['Default'],
        
        // Similarity score (can be calculated or set to default)
        similarityScore: 0,
        
        // Notes
        notes: generateComparisonNotes(existingProductData, goldStandardData, productAttributes),
        
        deliveryDate: `Delivery by ${getEstimatedDelivery()}`
      };
      
      setComparisonData(data);
      console.log('✅ Built comparison data:', data);
    } else {
      // Fallback to default data if no existing product
      console.warn('⚠️ No existing product found in localStorage, using default data');
      const defaultData = {
        originalTitle: "Product Not Found",
        originalPrice: 0,
        originalAsin: "N/A",
        originalImage: "https://placehold.co/400x400/E5E7EB/6B7280?text=No+Product",
        originalRating: 0,
        originalReviews: 0,
        
        competitorTitle: "No Gold Standard",
        competitorPrice: 0,
        competitorAsin: "N/A",
        competitorImage: "https://placehold.co/400x400/E5E7EB/6B7280?text=No+Image",
        competitorRating: 0,
        competitorReviews: 0,

        generatedImages: generatedImages || [],
        similarityScore: 0,
        notes: "No product data available. Please go back and select a product.",
        
        brandComparison: {
          original: "N/A",
          competitor: "N/A"
        },
        features: [],
        sizes: [],
        colors: [],
        deliveryDate: "N/A"
      };

      setComparisonData(defaultData);
    }
  }, [navigate]);

  // Helper functions to extract data from product
  const extractPrice = (product) => {
    if (!product) return 0;
    // Try to find price in various formats
    if (product.price) return parseFloat(product.price);
    if (product.productDetails?.Price) return parseFloat(product.productDetails.Price);
    if (product.productDetailsArray) {
      const priceDetail = product.productDetailsArray.find(d => 
        d.label && d.label.toLowerCase().includes('price')
      );
      if (priceDetail) return parseFloat(priceDetail.value);
    }
    return 0;
  };

  const extractRating = (product) => {
    if (!product) return 0;
    if (product.rating) return parseFloat(product.rating);
    if (product.productDetails?.Rating) return parseFloat(product.productDetails.Rating);
    return 0;
  };

  const extractReviews = (product) => {
    if (!product) return 0;
    if (product.reviews) return parseInt(product.reviews);
    if (product.reviewCount) return parseInt(product.reviewCount);
    if (product.productDetails?.Reviews) return parseInt(product.productDetails.Reviews);
    return 0;
  };

  const extractBrand = (product) => {
    if (!product) return null;
    if (product.brand) return product.brand;
    
    // Try to extract from manufacturing details
    if (product.manufacturingDetails?.Manufacturer) {
      const manufacturer = product.manufacturingDetails.Manufacturer;
      // Extract brand name (first part before comma)
      return manufacturer.split(',')[0].trim();
    }
    if (product.productDetailsArray) {
      const brandDetail = product.productDetailsArray.find(d => 
        d.label && d.label.toLowerCase().includes('brand')
      );
      if (brandDetail) return brandDetail.value;
    }

    if(product.productDetails?.Brand) {
      return product.productDetails.Brand;
    }

    
    
    return null;
  };

  const extractFeatures = (product) => {
    console.log('🔧 extractFeatures called with product:', product);
    if (!product) {
      console.log('⚠️ No product provided to extractFeatures');
      return [];
    }
    
    // If features array exists, use it (THIS IS THE PRIMARY SOURCE)
    if (product.features && Array.isArray(product.features) && product.features.length > 0) {
      console.log('✅ Found features array in product:', product.features);
      console.log('📊 Number of features:', product.features.length);
      return product.features;
    }
    
    console.log('⚠️ No features array found in product, trying fallback methods');
    
    // Try to extract from description
    const features = [];
    if (product.description) {
      features.push(product.description);
      console.log('📝 Added description as feature:', product.description);
    }
    
    // Add generic features from product details
    if (product.productDetails?.Department) {
      features.push(`Department: ${product.productDetails.Department}`);
    }
    if (product.productDetails?.['Generic Name']) {
      features.push(`Type: ${product.productDetails['Generic Name']}`);
    }
    
    console.log('🔚 Final features array:', features.length > 0 ? features : ['No features available']);
    return features.length > 0 ? features : ['No features available'];
  };

  const extractDepartment = (product) => {
    if (!product) return null;
    if (product.productDetails?.Department) return product.productDetails.Department;
    
    // Search in productDetailsArray
    if (product.productDetailsArray) {
      const deptDetail = product.productDetailsArray.find(d => 
        d.label && d.label.toLowerCase().includes('department')
      );
      if (deptDetail) return deptDetail.value;
    }
    
    return null;
  };

  const extractSizes = (product) => {
    if (!product) return null;
    // This would need to be extracted from product variations or details
    // For now, return null to use defaults
    return null;
  };

  const extractColors = (product) => {
    if (!product) return null;
    // This would need to be extracted from product variations or details
    // For now, return null to use defaults
    return null;
  };

  const generateComparisonNotes = (existingProduct, goldStandard, attributes) => {
    if (!existingProduct) return 'No product data available for comparison.';
    
    let notes = '';
    
    if (goldStandard) {
      notes += `Comparing "${existingProduct.title}" with gold standard "${goldStandard.title}". `;
    } else {
      notes += `Analyzing product: "${existingProduct.title}". `;
    }
    
    if (attributes && Object.keys(attributes).length > 0) {
      notes += `Product attributes: ${Object.entries(attributes).map(([k, v]) => `${k}: ${v}`).join(', ')}. `;
    }
    
    if (existingProduct.description) {
      notes += existingProduct.description;
    }
    
    return notes || 'No additional notes available.';
  };

  const getEstimatedDelivery = () => {
    const date = new Date();
    date.setDate(date.getDate() + 3); // 3 days from now
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  if (!comparisonData || !userInfo) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-orange-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <nav className="bg-[#131921] text-white px-4 py-2">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="text-2xl font-bold">Amazon</div>
          <div className="text-sm flex items-center gap-4">
            <span className="text-gray-300">Generated by {userInfo.name}</span>
            <button
              onClick={() => navigate('/seller-list')}
              className="bg-[#febd69] text-black px-4 py-1 rounded hover:bg-[#f3a847]"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </nav>

      <div className="bg-[#232f3e] text-white px-4 py-2 mb-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-lg text-center">
            Final Preview
          </div>
        </div>
      </div>

      <div ref={contentRef} className="max-w-7xl mx-auto p-4">

        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Image */}
          <div className="col-span-5">
            <div className="sticky top-4">
              {/* Main Image Display with Slider */}
              <div className="border p-4 mb-4 bg-white rounded-lg">
                <div className="h-[500px] flex items-center justify-center bg-gray-50 relative group rounded-lg overflow-hidden">
                  {allImages.length > 0 ? (
                    <>
                      <img
                        src={allImages[currentImageIndex]?.url}
                        alt={allImages[currentImageIndex]?.label || 'Product Image'}
                        className="max-h-full max-w-full object-contain"
                        onError={(e) => {
                          e.target.src = `https://placehold.co/500x500/E5E7EB/6B7280?text=Image+Error`;
                        }}
                      />
                      
                      {/* Navigation Arrows */}
                      {allImages.length > 1 && (
                        <>
                          <button
                            onClick={prevImage}
                            className="absolute left-2 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                            title="Previous Image"
                          >
                            <ChevronLeft className="h-6 w-6" />
                          </button>
                          <button
                            onClick={nextImage}
                            className="absolute right-2 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                            title="Next Image"
                          >
                            <ChevronRight className="h-6 w-6" />
                          </button>
                        </>
                      )}
                      
                      {/* Image Type Badge */}
                      <div className="absolute top-3 left-3">
                        {/* <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                          allImages[currentImageIndex]?.type === 'generated' 
                            ? 'bg-blue-500 text-white' 
                            : 'bg-green-500 text-white'
                        }`}>
                          {allImages[currentImageIndex]?.type === 'generated' ? '🎨 AI Generated' : '📦 Existing'}
                        </span> */}
                      </div>
                      
                      {/* Image Counter */}
                      <div className="absolute bottom-3 right-3 bg-black/70 text-white px-3 py-1 rounded-full text-sm">
                        {currentImageIndex + 1} / {allImages.length}
                      </div>
                    </>
                  ) : (
                    <div className="text-center text-gray-400">
                      <p className="text-lg mb-2">No images available</p>
                      <p className="text-sm">Please upload or generate product images</p>
                    </div>
                  )}
                </div>
                
                {/* Image Label */}
                <div className="text-center mt-3">
                  <p className="text-sm font-medium text-gray-900">
                    {allImages[currentImageIndex]?.label || 'Product Image'}
                  </p>
                </div>
              </div>

              {/* Thumbnail Strip */}
              {allImages.length > 0 && (
                <div className="border rounded-lg p-3 bg-white">
                  <div className="flex gap-2 overflow-x-auto pb-2">
                    {allImages.map((image, index) => (
                      <div
                        key={index}
                        onClick={() => goToImage(index)}
                        className={`flex-shrink-0 cursor-pointer border-2 rounded-lg overflow-hidden transition-all ${
                          currentImageIndex === index 
                            ? 'border-blue-500 ring-2 ring-blue-300' 
                            : 'border-gray-200 hover:border-blue-300'
                        }`}
                        style={{ width: '80px', height: '80px' }}
                      >
                        <div className="relative w-full h-full">
                          <img
                            src={image.url}
                            alt={image.label}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              e.target.src = `https://placehold.co/80x80/E5E7EB/6B7280?text=${index + 1}`;
                            }}
                          />
                          {/* Small badge for type */}
                          <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] text-center py-0.5">
                            {image.type === 'generated' ? 'AI' : 'Orig'}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  {/* <p className="text-xs text-gray-500 mt-2 text-center">
                    Click thumbnails to view • {allImages.filter(img => img.type === 'existing').length} Existing • {allImages.filter(img => img.type === 'generated').length} AI Generated
                  </p> */}
                </div>
              )}

              {/* <div className="bg-[#fef8f2] border border-[#f4d499] rounded p-3 mt-4">
                <div className="text-sm">
                  <span className="font-bold text-[#c45500]">Similarity Score: </span>
                  <span className="text-[#c45500]">{comparisonData?.similarityScore}% match with original product</span>
                </div>
              </div> */}
            </div>
          </div>

          {/* Right Column - Product Details */}
          <div className="col-span-7">
            <div className="border-b border-[#e7e7e7] pb-4">
              <h1 className="text-xl font-medium mb-2">
                {comparisonData?.originalTitle}
              </h1>
              {/* <div className="flex items-center gap-2 text-sm">
                <div className="text-[#0F1111]">
                  <span className="text-[#007185] hover:text-[#c45500] cursor-pointer">
                    {comparisonData?.brandComparison.original}
                  </span>
                </div>
                <span className="text-[#707070]">|</span>
                <div className="flex items-center gap-1">
                  <span className="text-[#f2c200]">{renderStars(comparisonData?.originalRating)}</span>
                  <span className="text-[#007185] hover:text-[#c45500] cursor-pointer">
                    {comparisonData?.originalReviews} ratings
                  </span>
                </div>
              </div> */}
            </div>

            {/* Price Section */}
            <div className="py-4 border-b border-[#e7e7e7]">
              <div className="text-[#0F1111]">
                <span className="text-sm">Price:</span>
                <span className="text-[28px] text-[#0F1111] ml-2">
                  ₹{comparisonData?.originalPrice}
                </span>
              </div>
              
              {comparisonData?.competitorPrice > 0 && (
                <div className="mt-1">
                  <div className="inline-flex items-center bg-[#fef8f2] px-2 py-1 rounded">
                    <span className="text-sm text-[#c45500]">
                      {comparisonData?.originalPrice < comparisonData?.competitorPrice 
                        ? 'Lower price than gold standard by '
                        : 'Higher price than gold standard by '}
                      ${Math.abs(comparisonData?.originalPrice - comparisonData?.competitorPrice).toFixed(2)}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Product Information */}
            <div className="py-4 border-b border-[#e7e7e7]">
              <table className="w-full text-sm">
                <tbody>
                  <tr>
                    <td className="py-1 pr-4 align-top text-[#565959]" style={{ width: '120px' }}>Brand</td>
                    <td className="py-1 text-[#0F1111]">{comparisonData?.brandComparison.original}</td>
                  </tr>
                  <tr>
                    <td className="py-1 pr-4 align-top text-[#565959]">ASIN</td>
                    <td className="py-1 text-[#0F1111]">{comparisonData?.originalAsin}</td>
                  </tr>
                  <tr>
                    {/* <td className="py-1 pr-4 align-top text-[#565959]">Rating</td> */}
                    {/* <td className="py-1 text-[#0F1111]">
                      {comparisonData?.originalRating} out of 5 ({comparisonData?.originalReviews} reviews)
                    </td> */}
                  </tr>
                  {comparisonData?.category && comparisonData.category !== 'N/A' && (
                    <tr>
                      <td className="py-1 pr-4 align-top text-[#565959]">Category</td>
                      <td className="py-1 text-[#0F1111]">{comparisonData.category}</td>
                    </tr>
                  )}
                  {comparisonData?.subcategory && comparisonData.subcategory !== 'N/A' && (
                    <tr>
                      <td className="py-1 pr-4 align-top text-[#565959]">Subcategory</td>
                      <td className="py-1 text-[#0F1111]">{comparisonData.subcategory}</td>
                    </tr>
                  )}
                  {comparisonData?.department && comparisonData.department !== 'General' && (
                    <tr>
                      <td className="py-1 pr-4 align-top text-[#565959]">Department</td>
                      <td className="py-1 text-[#0F1111]">{comparisonData.department}</td>
                    </tr>
                  )}
                </tbody>
              </table>

              {/* Product Attributes */}
              {/* {comparisonData?.productAttributes && Object.keys(comparisonData.productAttributes).length > 0 && (
                <div className="mt-3 p-3 bg-blue-50 rounded border border-blue-200">
                  <h4 className="text-xs font-semibold text-blue-900 mb-2">Product Attributes</h4>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(comparisonData.productAttributes).map(([key, value]) => (
                      <span key={key} className="inline-flex items-center bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                        <strong className="mr-1">{key}:</strong> {value}
                      </span>
                    ))}
                  </div>
                </div>
              )} */}
            </div>

              {/* Features */}
              <div className="border-b border-[#e7e7e7] pb-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-[#0F1111] font-medium text-lg">About this item</h3>                 
                </div>
                <ul className="list-disc pl-5 space-y-1 text-sm text-[#0F1111]">
                  {comparisonData?.features && comparisonData.features.length > 0 ? (
                    comparisonData.features.map((feature, index) => (
                      <li key={index}>{feature}</li>
                    ))
                  ) : (
                    <li>No features available</li>
                  )}
                </ul>
              </div>

              {/* Product Details */}
              {comparisonData?.productDetails && comparisonData.productDetails.length > 0 && (
                <div className="py-4 border-b border-[#e7e7e7]">
                  <h3 className="text-[#0F1111] font-medium text-lg mb-2">Product Details</h3>
                  <table className="w-full text-sm">
                    <tbody>
                      {comparisonData.productDetails.map((detail, index) => (
                        <tr key={index}>
                          <td className="py-1 pr-4 align-top text-[#565959]" style={{ width: '200px' }}>
                            {detail.label.replace(/[\n\s:‏‎]+/g, ' ').trim()}
                          </td>
                          <td className="py-1 text-[#0F1111]">{detail.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Manufacturing Details */}
              {comparisonData?.manufacturingDetails && comparisonData.manufacturingDetails.length > 0 && (
                <div className="py-4 border-b border-[#e7e7e7]">
                  <h3 className="text-[#0F1111] font-medium text-lg mb-2">Manufacturing Details</h3>
                  <table className="w-full text-sm">
                    <tbody>
                      {comparisonData.manufacturingDetails.slice(0, 5).map((detail, index) => (
                        <tr key={index}>
                          <td className="py-1 pr-4 align-top text-[#565959]" style={{ width: '200px' }}>
                            {detail.label.replace(/[\n\s:‏‎]+/g, ' ').trim()}
                          </td>
                          <td className="py-1 text-[#0F1111]">{detail.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Size & Color Options */}
              {((comparisonData?.sizes && comparisonData.sizes.length > 0) || 
                (comparisonData?.colors && comparisonData.colors.length > 0)) && (
                <div className="py-4 border-[#e7e7e7]">
                  {comparisonData?.sizes && comparisonData.sizes.length > 0 && (
                    <div className="mb-4">
                      {/* <span className="text-sm font-medium text-[#0F1111]">Size: </span>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {comparisonData.sizes.map((size, index) => (
                          <button
                            key={index}
                            className="min-w-[60px] px-3 py-1 border border-[#D5D9D9] rounded-lg text-sm
                            hover:border-[#007185] cursor-pointer"
                          >
                            {size}
                          </button>
                        ))}
                      </div> */}
                    </div>
                  )}
                  {/* {comparisonData?.colors && comparisonData.colors.length > 0 && (
                    <div>
                      <span className="text-sm font-medium text-[#0F1111]">Color: </span>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {comparisonData.colors.map((color, index) => (
                          <button
                            key={index}
                            className="min-w-[60px] px-3 py-1 border border-[#D5D9D9] rounded-lg text-sm
                            hover:border-[#007185] cursor-pointer"
                          >
                            {color}
                          </button>
                        ))}
                      </div>
                    </div>
                  )} */}
                </div>
              )}

              {/* Delivery & Analysis */}
              <div className="py-4 space-y-4">
                {/* <div className="text-[#007600] font-medium">
                  {comparisonData?.deliveryDate}
                </div> */}
                
                {/* <div className="bg-[#F4F4F4] p-4 rounded">
                  <h3 className="text-[#0F1111] font-medium mb-2">Comparison Analysis</h3>
                  <p className="text-sm text-[#333333] leading-relaxed">{comparisonData?.notes}</p>
                </div> */}

                {/* Action Buttons */}
                <div className="flex flex-col gap-3 mt-6">
                  <button
                    onClick={handleProceedAndExportToSheets}
                    disabled={isExporting}
                    className="w-full bg-[#FFD814] hover:bg-[#F7CA00] text-[#0F1111] py-3 px-4 rounded-lg 
                    shadow-sm border border-[#FCD200] font-medium text-base transition-colors
                    disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isExporting ? (
                      <>
                        <svg className="animate-spin h-5 w-5 text-[#0F1111]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Exporting to Google Sheets...</span>
                      </>
                    ) : (
                      'Export to Google Sheets'
                    )}
                  </button>
                  {/* <button
                    onClick={handleDownload}
                    className="w-full bg-[#F0F2F2] hover:bg-[#DFE1E1] text-[#0F1111] py-2 px-4 rounded-lg 
                    shadow-sm border border-[#D5D9D9]"
                  >
                    Download PDF Report
                  </button>
                  <button
                    onClick={() => window.print()}
                    className="w-full bg-[#F0F2F2] hover:bg-[#DFE1E1] text-[#0F1111] py-2 px-4 rounded-lg
                    shadow-sm border border-[#D5D9D9]"
                  >
                    Print Report
                  </button> */}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
  )};

export default FinalPage;
