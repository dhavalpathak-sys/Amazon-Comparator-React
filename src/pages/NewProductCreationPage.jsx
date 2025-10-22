// Page 4b: New Product Creation Page with Form and Gold Standard Reference

import { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { Star, Save, Plus, Check, Wand2, Edit2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { goldStandardProduct } from '../data/mockData.js';
import { generateProductTitle, generateProductDescription } from '../api/api.js';

const NewProductCreationPage = () => {
  const navigate = useNavigate();

  // Golden Product Data from Local Storage
  const [goldenProduct, setGoldenProduct] = useState(null);

  // Product Title State
  const [productTitle, setProductTitle] = useState('');
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [acceptedTitle, setAcceptedTitle] = useState('');
  const titleTextareaRef = useRef(null);
  
  // Product Description State
  const [productDescription, setProductDescription] = useState('');
  const [isEditingDescription, setIsEditingDescription] = useState(false);
  const [acceptedDescription, setAcceptedDescription] = useState('');
  const descTextareaRef = useRef(null);
  
  // Product Attributes State - will be dynamically initialized
  const [productAttributes, setProductAttributes] = useState({});

  const [showAddAttribute, setShowAddAttribute] = useState(false);
  const [newAttributeKey, setNewAttributeKey] = useState('');
  const [newAttributeValue, setNewAttributeValue] = useState('');
  const [selectedAttributeType, setSelectedAttributeType] = useState('');

  // Loading states for AI generation
  const [isGeneratingTitle, setIsGeneratingTitle] = useState(false);
  const [isGeneratingDescription, setIsGeneratingDescription] = useState(false);

  // Predefined attribute options
  const predefinedAttributes = [
    { key: 'material', label: 'Material' },
    { key: 'color', label: 'Color' },
    { key: 'size', label: 'Size' },
    { key: 'custom', label: 'Custom Attribute' }
  ];

  // Fetch Golden Product from Local Storage on Component Mount
  useEffect(() => {
    try {
      const storedGoldenProduct = localStorage.getItem('goldStandardProduct');
      if (storedGoldenProduct) {
        const parsedProduct = JSON.parse(storedGoldenProduct);
        setGoldenProduct(parsedProduct);
        
        // Initialize product attributes dynamically based on golden product
        // Combine all available attribute arrays (productDetails, manufacturing, additionalInfo)
        const initialAttributes = {};
        
        if (parsedProduct.details) {
          // MockData format with details object
          Object.keys(parsedProduct.details).forEach(key => {
            initialAttributes[key] = '';
          });
        } else {
          // Scraped format - combine all attribute arrays
          const allArrays = [
            ...(parsedProduct.productDetailsArray || []),
            ...(parsedProduct.manufacturingDetailsArray || []),
            ...(parsedProduct.additionalInfoArray || [])
          ];
          
          allArrays.forEach(item => {
            const key = item.label?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '') || 
                       item.key?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
            if (key && !initialAttributes[key]) { // Avoid duplicates
              initialAttributes[key] = '';
            }
          });
          
          // If no attributes found, use mockData structure
          if (Object.keys(initialAttributes).length === 0) {
            Object.keys(goldStandardProduct.details).forEach(key => {
              initialAttributes[key] = '';
            });
          }
        }
        
        setProductAttributes(initialAttributes);
      } else {
        // Fallback to mockData if nothing in localStorage
        setGoldenProduct(goldStandardProduct);
        
        // Initialize attributes from mockData
        const initialAttributes = {};
        Object.keys(goldStandardProduct.details).forEach(key => {
          initialAttributes[key] = '';
        });
        setProductAttributes(initialAttributes);
      }
    } catch (error) {
      console.error('Error fetching golden product from localStorage:', error);
      // Fallback to mockData
      setGoldenProduct(goldStandardProduct);
      
      const initialAttributes = {};
      Object.keys(goldStandardProduct.details).forEach(key => {
        initialAttributes[key] = '';
      });
      setProductAttributes(initialAttributes);
    }
  }, []);

  const handleAttributeChange = (key, value) => {
    setProductAttributes(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleAddAttribute = () => {
    const attributeKey = selectedAttributeType === 'custom' 
      ? newAttributeKey.trim().toLowerCase().replace(/\s+/g, '') 
      : selectedAttributeType;
    
    if (attributeKey && newAttributeValue.trim()) {
      setProductAttributes(prev => ({
        ...prev,
        [attributeKey]: newAttributeValue.trim()
      }));
      setNewAttributeKey('');
      setNewAttributeValue('');
      setSelectedAttributeType('');
      setShowAddAttribute(false);
    }
  };

  const handleSaveDraft = () => {
    // Build product object similar to existingProduct structure
    const newProductData = {
      id: 'NEW-' + Date.now(), // Generate unique ID
      asin: 'NEW-' + Date.now(),
      title: acceptedTitle,
      brand: null,
      amazonUrl: null, // No URL for new products
      
      // Images - empty for now (will be generated)
      images: [],
      
      // Features/Bullets - use description as a feature
      features: acceptedDescription ? [acceptedDescription] : [],
      
      // Description
      description: acceptedDescription,
      
      // Convert attributes to array format matching existing product structure
      productDetailsArray: Object.entries(productAttributes).map(([key, value]) => ({
        label: key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1').trim(),
        key: key,
        value: value || 'N/A'
      })),
      
      manufacturingDetailsArray: [],
      additionalInfoArray: [],
      
      // Mark as new product
      isNewProduct: true
    };
    
    console.log('💾 Saving new product data:', newProductData);
    
    // Save to SAME localStorage key as existing product
    localStorage.setItem('existingProduct', JSON.stringify(newProductData));
    console.log('✅ Stored new product in existingProduct localStorage');
    
    // Navigate with isNewProduct flag to show 2-column layout
    navigate('/compare-images', { state: { isNewProduct: true } });
  };

  // Auto-resize textareas
  useLayoutEffect(() => {
    if (titleTextareaRef.current) {
      titleTextareaRef.current.style.height = 'auto';
      titleTextareaRef.current.style.height = titleTextareaRef.current.scrollHeight + 'px';
    }
  }, [productTitle]);

  useLayoutEffect(() => {
    if (descTextareaRef.current) {
      descTextareaRef.current.style.height = 'auto';
      descTextareaRef.current.style.height = descTextareaRef.current.scrollHeight + 'px';
    }
  }, [productDescription]);

  // Title Handlers
  const handleAcceptTitle = () => {
    setAcceptedTitle(productTitle);
    setIsEditingTitle(false);
  };

  const handleCancelTitle = () => {
    setProductTitle(acceptedTitle);
    setIsEditingTitle(false);
  };

  const handleAIGenerateTitle = async () => {
    setIsGeneratingTitle(true);
    try {
      const subcategory = 'Product'; // You can make this dynamic
      
      // Clean product attributes before sending - create readable labels
      const cleanedProductDetails = {};
      Object.entries(productAttributes).forEach(([key, value]) => {
        if (value && value.trim()) {
          // Convert camelCase or lowercase keys to readable labels
          const readableKey = key
            .replace(/([A-Z])/g, ' $1') // Add space before capital letters
            .replace(/([a-z])([A-Z])/g, '$1 $2') // Add space between lower and upper
            .trim()
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          
          cleanedProductDetails[readableKey] = value.trim();
        }
      });

      if (Object.keys(cleanedProductDetails).length === 0) {
        alert('Please fill in some product attributes before generating AI title.');
        setIsGeneratingTitle(false);
        return;
      }

      console.log('🎯 Sending to AI (Title):', cleanedProductDetails);

      const generatedTitle = await generateProductTitle(subcategory, cleanedProductDetails);
      setProductTitle(generatedTitle);
      setIsEditingTitle(true);
    } catch (error) {
      console.error('Error generating title:', error);
      alert(`Failed to generate title: ${error.message}`);
    } finally {
      setIsGeneratingTitle(false);
    }
  };

  // Description Handlers
  const handleAcceptDescription = () => {
    setAcceptedDescription(productDescription);
    setIsEditingDescription(false);
  };

  const handleCancelDescription = () => {
    setProductDescription(acceptedDescription);
    setIsEditingDescription(false);
  };

  const handleAIGenerateDescription = async () => {
    setIsGeneratingDescription(true);
    try {
      const subcategory = 'Product'; // You can make this dynamic
      
      // Clean product attributes before sending - create readable labels
      const cleanedProductDetails = {};
      Object.entries(productAttributes).forEach(([key, value]) => {
        if (value && value.trim()) {
          // Convert camelCase or lowercase keys to readable labels
          const readableKey = key
            .replace(/([A-Z])/g, ' $1') // Add space before capital letters
            .replace(/([a-z])([A-Z])/g, '$1 $2') // Add space between lower and upper
            .trim()
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          
          cleanedProductDetails[readableKey] = value.trim();
        }
      });

      if (Object.keys(cleanedProductDetails).length === 0) {
        alert('Please fill in some product attributes before generating AI description.');
        setIsGeneratingDescription(false);
        return;
      }

      console.log('🎯 Sending to AI (Description):', cleanedProductDetails);

      const generatedDescription = await generateProductDescription(subcategory, cleanedProductDetails);
      setProductDescription(generatedDescription);
      setIsEditingDescription(true);
    } catch (error) {
      console.error('Error generating description:', error);
      alert(`Failed to generate description: ${error.message}`);
    } finally {
      setIsGeneratingDescription(false);
    }
  };

  const calculateProgress = () => {
    // Count filled fields dynamically
    let totalFields = 0;
    let filledFields = 0;

    // 1. Title field
    totalFields += 1;
    if (productTitle.trim() !== '') filledFields += 1;

    // 2. Description field
    totalFields += 1;
    if (productDescription.trim() !== '') filledFields += 1;

    // 3. All attribute fields (including custom attributes)
    // Simply count all attributes in productAttributes state
    Object.entries(productAttributes).forEach(([key, value]) => {
      totalFields += 1;
      if (value && value.trim() !== '') {
        filledFields += 1;
      }
    });

    return { filled: filledFields, total: totalFields };
  };

  // Helper function to get description from golden product
  const getDescription = () => {
    if (!goldenProduct) return '';
    // Try different possible description fields
    return goldenProduct.description || 
           (goldenProduct.features && goldenProduct.features.length > 0 ? goldenProduct.features.join(' • ') : '') ||
           'No description available';
  };

  // Helper function to get all attribute sections
  const getAllAttributeSections = () => {
    if (!goldenProduct) return [];
    
    const sections = [];
    
    // Product Details
    if (goldenProduct.productDetailsArray && goldenProduct.productDetailsArray.length > 0) {
      sections.push({
        title: 'Product Details',
        color: 'blue',
        items: goldenProduct.productDetailsArray
      });
    }
    
    // Manufacturing Details
    if (goldenProduct.manufacturingDetailsArray && goldenProduct.manufacturingDetailsArray.length > 0) {
      sections.push({
        title: 'Manufacturing Details',
        color: 'green',
        items: goldenProduct.manufacturingDetailsArray
      });
    }
    
    // Additional Info
    if (goldenProduct.additionalInfoArray && goldenProduct.additionalInfoArray.length > 0) {
      sections.push({
        title: 'Additional Information',
        color: 'purple',
        items: goldenProduct.additionalInfoArray
      });
    }
    
    return sections;
  };

  const progress = calculateProgress();
  const isFormComplete = progress.filled === progress.total;

  // Show loading state while fetching golden product
  if (!goldenProduct) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading golden product data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6">
        {/* Page Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Create New Product</h1>
          <p className="text-sm text-gray-600">
            Use the Gold Standard as your guide to create an optimized product listing
          </p>
        </div>

        {/* Main Content Container */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          
          {/* Column Headers */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6 pb-4 border-b border-gray-200">
            {/* Left Column Header */}
            <div className="flex items-center gap-2">
              <Star className="h-6 w-6 fill-yellow-400 text-yellow-400" />
              <h2 className="text-xl font-semibold text-gray-900">Gold Standard Product</h2>
            </div>
            {/* Right Column Header */}
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Your New Product</h2>
            </div>
          </div>

          {/* Product Attributes Section - MOVED TO TOP */}
          <div className="mb-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
              <h3 className="text-lg font-semibold text-gray-700">Product Attributes</h3>
              <h3 className="text-lg font-semibold text-gray-700">Product Attributes</h3>
            </div>
            <div className="space-y-3">
              {goldenProduct.details ? (
                // MockData format - side by side display
                <>
                  {Object.entries(productAttributes).map(([key, value]) => (
                    <div key={key} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Left: Gold Standard Attribute */}
                      <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-3">
                        <div className="text-xs font-semibold text-gray-600 mb-2 capitalize">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </div>
                        <p className="text-sm font-medium text-gray-800">
                          {goldenProduct.details[key]}
                        </p>
                      </div>

                      {/* Right: Your Product Attribute Input */}
                      <div className="bg-white border border-gray-200 rounded-lg p-3">
                        <div className="text-xs font-semibold text-gray-600 mb-2 capitalize">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </div>
                        <input
                          type="text"
                          className="w-full border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          value={value}
                          onChange={e => handleAttributeChange(key, e.target.value)}
                          placeholder={`Enter ${key.replace(/([A-Z])/g, ' $1').trim().toLowerCase()}...`}
                        />
                      </div>
                    </div>
                  ))}
                </>
              ) : (
                // Scraped format with multiple arrays - side by side display
                <>
                  {(() => {
                    const allAttributes = [];
                    const seenKeys = new Set();
                    const normalizeKey = (key) => {
                      if (!key) return '';
                      return key.trim()
                        .replace(/[\n\r\s:‏‎]+/g, '')
                        .toLowerCase();
                    };

                    // Collect all attributes from all sections with deduplication
                    if (goldenProduct.productDetailsArray && goldenProduct.productDetailsArray.length > 0) {
                      goldenProduct.productDetailsArray.forEach((item) => {
                        const key = item.label?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
                        const normalizedKey = normalizeKey(item.label || item.key || '');
                        if (!seenKeys.has(normalizedKey)) {
                          seenKeys.add(normalizedKey);
                          allAttributes.push({ ...item, inputKey: key });
                        }
                      });
                    }
                    
                    if (goldenProduct.manufacturingDetailsArray && goldenProduct.manufacturingDetailsArray.length > 0) {
                      goldenProduct.manufacturingDetailsArray.forEach((item) => {
                        const key = item.label?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
                        const normalizedKey = normalizeKey(item.label || item.key || '');
                        if (!seenKeys.has(normalizedKey)) {
                          seenKeys.add(normalizedKey);
                          allAttributes.push({ ...item, inputKey: key });
                        }
                      });
                    }
                    
                    if (goldenProduct.additionalInfoArray && goldenProduct.additionalInfoArray.length > 0) {
                      goldenProduct.additionalInfoArray.forEach((item) => {
                        const key = item.label?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
                        const normalizedKey = normalizeKey(item.label || item.key || '');
                        if (!seenKeys.has(normalizedKey)) {
                          seenKeys.add(normalizedKey);
                          allAttributes.push({ ...item, inputKey: key });
                        }
                      });
                    }

                    // Render deduplicated attributes side by side
                    return allAttributes.map((item, index) => (
                      <div key={`attr-${index}`} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Left: Gold Standard Attribute */}
                        <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-3">
                          <div className="text-xs font-semibold text-gray-600 mb-2">
                            {item.label || item.key || 'Attribute'}
                          </div>
                          <p className="text-sm font-medium text-gray-800">{item.value || 'N/A'}</p>
                        </div>

                        {/* Right: Your Product Attribute Input */}
                        <div className="bg-white border border-gray-200 rounded-lg p-3">
                          <div className="text-xs font-semibold text-gray-600 mb-2">
                            {item.label || item.key || 'Attribute'}
                          </div>
                          <input
                            type="text"
                            className="w-full border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            value={productAttributes[item.inputKey] || ''}
                            onChange={e => handleAttributeChange(item.inputKey, e.target.value)}
                            placeholder={`Enter ${item.label?.toLowerCase() || 'value'}...`}
                          />
                        </div>
                      </div>
                    ));
                  })()}
                  
                  {/* Custom Added Attributes Section */}
                  {(() => {
                    // Get all keys from golden product
                    const goldenKeys = new Set();
                    
                    if (goldenProduct.productDetailsArray) {
                      goldenProduct.productDetailsArray.forEach(item => {
                        const key = item.label?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
                        if (key) goldenKeys.add(key);
                      });
                    }
                    
                    if (goldenProduct.manufacturingDetailsArray) {
                      goldenProduct.manufacturingDetailsArray.forEach(item => {
                        const key = item.label?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
                        if (key) goldenKeys.add(key);
                      });
                    }
                    
                    if (goldenProduct.additionalInfoArray) {
                      goldenProduct.additionalInfoArray.forEach(item => {
                        const key = item.label?.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
                        if (key) goldenKeys.add(key);
                      });
                    }
                    
                    // Find custom attributes (not in golden product)
                    const customAttributes = Object.entries(productAttributes).filter(([key]) => !goldenKeys.has(key));
                    
                    return customAttributes.length > 0 && (
                      <>
                        {customAttributes.map(([key, value]) => (
                          <div key={`custom-${key}`} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Left: Empty placeholder */}
                            <div className="bg-gray-100 border border-gray-300 rounded-lg p-3">
                              <div className="text-xs font-semibold text-gray-600 mb-2 capitalize">
                                {key.replace(/([A-Z])/g, ' $1').trim()}
                              </div>
                              <p className="text-sm text-gray-400 italic">Custom attribute</p>
                            </div>
                            
                            {/* Right: Custom Attribute Input */}
                            <div className="bg-white border border-gray-200 rounded-lg p-3">
                              <div className="text-xs font-semibold text-gray-600 mb-2 capitalize">
                                {key.replace(/([A-Z])/g, ' $1').trim()}
                              </div>
                              <input
                                type="text"
                                className="w-full border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                value={value}
                                onChange={e => handleAttributeChange(key, e.target.value)}
                                placeholder={`Enter ${key.replace(/([A-Z])/g, ' $1').trim().toLowerCase()}...`}
                              />
                            </div>
                          </div>
                        ))}
                      </>
                    );
                  })()}
                </>
              )}
              
              {/* Add Attribute Button/Form */}
              {showAddAttribute ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div></div>
                  <div className="bg-blue-50 border border-blue-300 rounded-lg p-3">
                    <div className="space-y-2">
                      {/* Dropdown for attribute selection */}
                      <select
                        className="border border-gray-300 rounded-md px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        value={selectedAttributeType}
                        onChange={e => setSelectedAttributeType(e.target.value)}
                      >
                        <option value="">Select Attribute Type</option>
                        {predefinedAttributes.map(attr => (
                          <option key={attr.key} value={attr.key}>
                            {attr.label}
                          </option>
                        ))}
                      </select>
                      
                      {/* Show custom attribute name input only if "Custom Attribute" is selected */}
                      {selectedAttributeType === 'custom' && (
                        <input
                          type="text"
                          placeholder="Custom Attribute Name (e.g., Brand)"
                          className="border border-gray-300 rounded-md px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          value={newAttributeKey}
                          onChange={e => setNewAttributeKey(e.target.value)}
                        />
                      )}
                      
                      {/* Attribute value input */}
                      <input
                        type="text"
                        placeholder={`Enter ${selectedAttributeType === 'custom' ? 'attribute' : selectedAttributeType || 'attribute'} value...`}
                        className="border border-gray-300 rounded-md px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        value={newAttributeValue}
                        onChange={e => setNewAttributeValue(e.target.value)}
                        disabled={!selectedAttributeType}
                      />
                      
                      <div className="flex gap-2">
                        <button
                          className={`flex items-center space-x-1 px-3 py-1.5 rounded-md transition-colors text-sm ${
                            selectedAttributeType && newAttributeValue.trim() && (selectedAttributeType !== 'custom' || newAttributeKey.trim())
                              ? 'bg-green-600 text-white hover:bg-green-700'
                              : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                          }`}
                          onClick={handleAddAttribute}
                          disabled={!selectedAttributeType || !newAttributeValue.trim() || (selectedAttributeType === 'custom' && !newAttributeKey.trim())}
                        >
                          <Check className="h-3.5 w-3.5" />
                          <span>Add</span>
                        </button>
                        <button
                          className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors text-sm"
                          onClick={() => {
                            setShowAddAttribute(false);
                            setNewAttributeKey('');
                            setNewAttributeValue('');
                            setSelectedAttributeType('');
                          }}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div></div>
                  <button
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm justify-center"
                    onClick={() => setShowAddAttribute(true)}
                  >
                    <Plus className="h-4 w-4" />
                    <span>Add New Attribute</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Product Title Section - MOVED BELOW ATTRIBUTES */}
          <div className="mb-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
              <h3 className="text-lg font-semibold text-gray-700">Product Title</h3>
              <h3 className="text-lg font-semibold text-gray-700">Product Title</h3>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Gold Standard Product Title */}
              <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-4 h-full flex items-center">
                <div className="bg-white/60 rounded-md p-3 border border-yellow-200 w-full">
                  <p className="text-gray-800 text-sm leading-relaxed">
                    {goldenProduct.title}
                  </p>
                </div>
              </div>

              {/* Right: Your Product Title Input with AI Generation */}
              <div className="h-full flex flex-col">
                {isEditingTitle ? (
                  <div className="space-y-3">
                    <textarea
                      ref={titleTextareaRef}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                      value={productTitle}
                      onChange={e => setProductTitle(e.target.value)}
                      rows={3}
                      placeholder="Enter your product title..."
                    />
                    <div className="flex gap-2">
                      <button
                        className="flex items-center space-x-1 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
                        onClick={handleAcceptTitle}
                      >
                        <Check className="h-4 w-4" />
                        <span>Accept</span>
                      </button>
                      <button
                        className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors text-sm"
                        onClick={handleCancelTitle}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-gray-50 min-h-[60px]">
                      {acceptedTitle || <span className="text-gray-400 italic">No title yet</span>}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        className="flex items-center space-x-1 px-3 py-1.5 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={handleAIGenerateTitle}
                        disabled={isGeneratingTitle}
                      >
                        <Wand2 className={`h-4 w-4 ${isGeneratingTitle ? 'animate-spin' : ''}`} />
                        <span>{isGeneratingTitle ? 'Generating...' : 'AI Generate'}</span>
                      </button>
                      <button
                        className="flex items-center space-x-1 px-3 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm"
                        onClick={() => setIsEditingTitle(true)}
                      >
                        <Edit2 className="h-4 w-4" />
                        <span>Edit</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Product Description Section */}
          <div className="mb-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
              <h3 className="text-lg font-semibold text-gray-700">Product Description</h3>
              <h3 className="text-lg font-semibold text-gray-700">Product Description</h3>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Gold Standard Description */}
              <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-4 h-full flex items-center">
                <div className="bg-white/60 rounded-md p-3 border border-yellow-200 w-full">
                  <p className="text-gray-800 text-sm leading-relaxed">
                    {getDescription()}
                  </p>
                </div>
              </div>

              {/* Right: Your Product Description Input with AI Generation */}
              <div className="h-full flex flex-col">
                {isEditingDescription ? (
                  <div className="space-y-3">
                    <textarea
                      ref={descTextareaRef}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                      value={productDescription}
                      onChange={e => setProductDescription(e.target.value)}
                      rows={4}
                      placeholder="Enter your product description..."
                    />
                    <div className="flex gap-2">
                      <button
                        className="flex items-center space-x-1 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
                        onClick={handleAcceptDescription}
                      >
                        <Check className="h-4 w-4" />
                        <span>Accept</span>
                      </button>
                      <button
                        className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors text-sm"
                        onClick={handleCancelDescription}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-gray-50 min-h-[80px]">
                      {acceptedDescription || <span className="text-gray-400 italic">No description yet</span>}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        className="flex items-center space-x-1 px-3 py-1.5 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={handleAIGenerateDescription}
                        disabled={isGeneratingDescription}
                      >
                        <Wand2 className={`h-4 w-4 ${isGeneratingDescription ? 'animate-spin' : ''}`} />
                        <span>{isGeneratingDescription ? 'Generating...' : 'AI Generate'}</span>
                      </button>
                      <button
                        className="flex items-center space-x-1 px-3 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm"
                        onClick={() => setIsEditingDescription(true)}
                      >
                        <Edit2 className="h-4 w-4" />
                        <span>Edit</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

        </div>

        {/* Action Bar */}
        <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              <span className="font-medium">Progress:</span>
              <span className="ml-2">{progress.filled} of {progress.total} fields completed</span>
            </div>
            <button
              onClick={handleSaveDraft}
              disabled={!isFormComplete}
              className={`flex items-center space-x-2 px-6 py-2 rounded-md transition-colors text-sm font-medium ${
                isFormComplete 
                  ? 'bg-green-600 text-white hover:bg-green-700 cursor-pointer' 
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              <Save className="h-4 w-4" />
              <span>Proceed</span>
            </button>
          </div>
        </div>

        {/* Tips Section */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-2 text-sm flex items-center gap-2">
            <span>💡</span> Optimization Tips
          </h4>
          <ul className="text-sm text-blue-800 space-y-1.5">
            <li>• Use relevant keywords in your title for better searchability</li>
            <li>• Include specific product details and benefits in the description</li>
            <li>• Fill in all attributes to provide complete product information</li>
            <li>• Reference the gold standard for inspiration on structure and content</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default NewProductCreationPage;