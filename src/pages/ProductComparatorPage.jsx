// Page 4a: Product Comparator Page with Editable Features with golden standard reference

import React, { useState, useRef, useLayoutEffect, useEffect } from 'react';
import { Wand2, Check, Edit2, Eye, ExternalLink, Star, Save, Plus, Loader2 } from 'lucide-react';
import { goldStandardProduct } from '../data/mockData.js';
import { useNavigate } from 'react-router-dom';
import { generateProductTitle, generateProductDescription, scrapeProductFromUrl } from '../api/api.js';

// Component for editable attributes (not features - for product details like brand, fabric, etc.)
function AttributeEditableCell({ attributeKey, initialValue, onAccept, isAccepted }) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(initialValue);
  const [acceptedValue, setAcceptedValue] = useState(initialValue);

  const handleAccept = () => {
    setAcceptedValue(value);
    setIsEditing(false);
    onAccept();
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 hover:border-blue-300 transition-all">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-semibold text-gray-600">{attributeKey}</div>
        {isAccepted && (
          <div className="flex items-center text-green-600">
            <Check className="h-4 w-4" />
          </div>
        )}
      </div>
      {isEditing ? (
        <div className="space-y-2">
          <input
            type="text"
            className="border border-gray-300 rounded-md px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={value}
            onChange={e => setValue(e.target.value)}
            autoFocus
          />
          <div className="flex gap-2">
            <button
              className="flex items-center space-x-1 px-3 py-1.5 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
              onClick={handleAccept}
            >
              <Check className="h-3.5 w-3.5" />
              <span>Accept</span>
            </button>
            <button
              className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors text-sm"
              onClick={() => {
                setValue(acceptedValue);
                setIsEditing(false);
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-between gap-3">
          <p className="text-gray-800 text-sm flex-1">{acceptedValue}</p>
          <div className="flex gap-2 flex-shrink-0">
            <button
              className="flex items-center space-x-1 px-2 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-xs"
              onClick={() => setIsEditing(true)}
            >
              <Edit2 className="h-3 w-3" />
              <span>Edit</span>
            </button>
            {!isAccepted && (
              <button
                className="flex items-center space-x-1 px-2 py-1 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-xs"
                onClick={handleAccept}
              >
                <Check className="h-3 w-3" />
                <span>Accept</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const ProductComparator = () => {
  const navigate = useNavigate();
  
  // Golden Product Data from Local Storage
  const [goldenProduct, setGoldenProduct] = useState(null);
  
  // URL Input State
  const [productUrl, setProductUrl] = useState('');
  const [isLoadingUrl, setIsLoadingUrl] = useState(false);
  const [scrapedData, setScrapedData] = useState(null);
  
  // Product Title State
  const [productTitle, setProductTitle] = useState('');
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [acceptedTitle, setAcceptedTitle] = useState('');
  const [originalTitle, setOriginalTitle] = useState('');
  const titleTextareaRef = useRef(null);

  // Product Description State
  const [productDescription, setProductDescription] = useState('');
  const [isEditingDescription, setIsEditingDescription] = useState(false);
  const [acceptedDescription, setAcceptedDescription] = useState('');
  const [originalDescription, setOriginalDescription] = useState('');
  const descTextareaRef = useRef(null);

  // Product Attributes State (details like brand, fabric, etc.)
  const [productAttributes, setProductAttributes] = useState([]);
  const [editingAttributes, setEditingAttributes] = useState({});
  const [attributeValues, setAttributeValues] = useState({});

  const [showAddAttribute, setShowAddAttribute] = useState(false);
  const [newAttributeKey, setNewAttributeKey] = useState('');
  const [newAttributeValue, setNewAttributeValue] = useState('');
  const [selectedAttributeType, setSelectedAttributeType] = useState('');

  // Predefined attribute options
  const predefinedAttributes = [
    { key: 'material', label: 'Material' },
    { key: 'color', label: 'Color' },
    { key: 'size', label: 'Size' },
    { key: 'custom', label: 'Custom Attribute' }
  ];

  // Loading states for AI generation
  const [isGeneratingTitle, setIsGeneratingTitle] = useState(false);
  const [isGeneratingDescription, setIsGeneratingDescription] = useState(false);

  // Fetch Golden Product from Local Storage on Component Mount
  useEffect(() => {
    try {
      const storedGoldenProduct = localStorage.getItem('goldStandardProduct');
      if (storedGoldenProduct) {
        const parsedProduct = JSON.parse(storedGoldenProduct);
        setGoldenProduct(parsedProduct);
        
        // Initialize title and description from golden product
        const title = parsedProduct.title || '';
        const description = parsedProduct.description || 
                          (parsedProduct.features && parsedProduct.features.length > 0 ? parsedProduct.features.join(' • ') : '') ||
                          '';
        
        setProductTitle(title);
        setAcceptedTitle(title);
        setOriginalTitle(title);
        
        setProductDescription(description);
        setAcceptedDescription(description);
        setOriginalDescription(description);
        
        // Initialize product attributes from all available sections
        const attributes = [];
        let attrId = 1;
        
        if (parsedProduct.details) {
          // MockData format
          Object.entries(parsedProduct.details).forEach(([key, value]) => {
            attributes.push({
              id: attrId++,
              key: key.replace(/([A-Z])/g, ' $1').trim(),
              value: value,
              accepted: false
            });
          });
        } else {
          // Scraped format - combine all attribute arrays
          if (parsedProduct.productDetailsArray) {
            parsedProduct.productDetailsArray.forEach(item => {
              attributes.push({
                id: attrId++,
                key: item.label || item.key || 'Attribute',
                value: item.value || '',
                accepted: false,
                section: 'Product Details'
              });
            });
          }
          
          if (parsedProduct.manufacturingDetailsArray) {
            parsedProduct.manufacturingDetailsArray.forEach(item => {
              // Check for duplicates (like ASIN)
              const existingAttr = attributes.find(attr => 
                attr.key === (item.label || item.key)
              );
              if (!existingAttr) {
                attributes.push({
                  id: attrId++,
                  key: item.label || item.key || 'Attribute',
                  value: item.value || '',
                  accepted: false,
                  section: 'Manufacturing Details'
                });
              }
            });
          }
          
          if (parsedProduct.additionalInfoArray && parsedProduct.additionalInfoArray.length > 0) {
            parsedProduct.additionalInfoArray.forEach(item => {
              attributes.push({
                id: attrId++,
                key: item.label || item.key || 'Attribute',
                value: item.value || '',
                accepted: false,
                section: 'Additional Information'
              });
            });
          }
        }
        
        setProductAttributes(attributes);
      } else {
        // Fallback to mockData if nothing in localStorage
        setGoldenProduct(goldStandardProduct);
        
        setProductTitle(goldStandardProduct.title);
        setAcceptedTitle(goldStandardProduct.title);
        setOriginalTitle(goldStandardProduct.title);
        
        setProductDescription(goldStandardProduct.description);
        setAcceptedDescription(goldStandardProduct.description);
        setOriginalDescription(goldStandardProduct.description);
        
        // Initialize from mockData
        const attributes = [];
        let attrId = 1;
        Object.entries(goldStandardProduct.details).forEach(([key, value]) => {
          attributes.push({
            id: attrId++,
            key: key.replace(/([A-Z])/g, ' $1').trim(),
            value: value,
            accepted: false
          });
        });
        setProductAttributes(attributes);
      }
    } catch (error) {
      console.error('Error fetching golden product from localStorage:', error);
      // Fallback to mockData
      setGoldenProduct(goldStandardProduct);
      
      setProductTitle(goldStandardProduct.title);
      setAcceptedTitle(goldStandardProduct.title);
      setOriginalTitle(goldStandardProduct.title);
      
      setProductDescription(goldStandardProduct.description);
      setAcceptedDescription(goldStandardProduct.description);
      setOriginalDescription(goldStandardProduct.description);
      
      const attributes = [];
      let attrId = 1;
      Object.entries(goldStandardProduct.details).forEach(([key, value]) => {
        attributes.push({
          id: attrId++,
          key: key.replace(/([A-Z])/g, ' $1').trim(),
          value: value,
          accepted: false
        });
      });
      setProductAttributes(attributes);
    }
  }, []);

  // Check if all attributes are accepted
  const allAttributesAccepted = productAttributes.every(attr => attr.accepted);

  // Handler for loading URL
  const handleLoadUrl = async () => {
    if (!productUrl.trim()) {
      alert('Please enter a valid product URL');
      return;
    }

    setIsLoadingUrl(true);
    try {
      const data = await scrapeProductFromUrl(productUrl);
      console.log('🔍 Scraped Product Data:', data);
      setScrapedData(data);
      
      // Set title and description from scraped data
      setProductTitle(data.title || '');
      setAcceptedTitle(data.title || '');
      setOriginalTitle(data.title || '');
      
      const description = data.description || 
                         (data.features && data.features.length > 0 ? data.features.join(' • ') : '') ||
                         '';
      setProductDescription(description);
      setAcceptedDescription(description);
      setOriginalDescription(description);
      
      // Initialize attribute values for editing
      const alignedAttrs = getAlignedAttributesFromData(data);
      const initialValues = {};
      const initialAccepted = {};
      alignedAttrs.forEach(attr => {
        initialValues[attr.id] = attr.scrapedValue;
        initialAccepted[attr.id] = false;
      });
      setAttributeValues(initialValues);
      setProductAttributes(alignedAttrs.map(attr => ({
        ...attr,
        accepted: false
      })));
      
      alert('Product data scraped successfully! Check console for details.');
    } catch (error) {
      console.error('Error scraping URL:', error);
      alert(`Failed to scrape product: ${error.message}`);
    } finally {
      setIsLoadingUrl(false);
    }
  };

  // Helper to get aligned attributes from scraped data
  const getAlignedAttributesFromData = (data) => {
    if (!goldenProduct) return [];

    const goldenAttrs = [];
    const scrapedAttrs = [];
    
    // Get golden product attributes
    if (goldenProduct.details) {
      Object.entries(goldenProduct.details).forEach(([key, value]) => {
        goldenAttrs.push({
          key: key.replace(/([A-Z])/g, ' $1').trim(),
          value: value,
          section: 'Product Details'
        });
      });
    } else {
      if (goldenProduct.productDetailsArray) {
        goldenProduct.productDetailsArray.forEach(item => {
          goldenAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Product Details'
          });
        });
      }
      
      if (goldenProduct.manufacturingDetailsArray) {
        goldenProduct.manufacturingDetailsArray.forEach(item => {
          goldenAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Manufacturing Details'
          });
        });
      }
      
      if (goldenProduct.additionalInfoArray) {
        goldenProduct.additionalInfoArray.forEach(item => {
          goldenAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Additional Information'
          });
        });
      }
    }

    // Get scraped product attributes
    if (data) {
      if (data.productDetailsArray) {
        data.productDetailsArray.forEach(item => {
          scrapedAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Product Details'
          });
        });
      }
      
      if (data.manufacturingDetailsArray) {
        data.manufacturingDetailsArray.forEach(item => {
          scrapedAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Manufacturing Details'
          });
        });
      }
      
      if (data.additionalInfoArray) {
        data.additionalInfoArray.forEach(item => {
          scrapedAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Additional Information'
          });
        });
      }
    }

    // Create aligned list
    const aligned = [];
    const scrapedMap = new Map();
    
    scrapedAttrs.forEach(attr => {
      scrapedMap.set(attr.key.toLowerCase(), attr);
    });

    goldenAttrs.forEach((goldenAttr, index) => {
      const scrapedAttr = scrapedMap.get(goldenAttr.key.toLowerCase());
      aligned.push({
        id: index,
        key: goldenAttr.key,
        goldenValue: goldenAttr.value,
        scrapedValue: scrapedAttr ? scrapedAttr.value : '',
        section: goldenAttr.section,
        isMatched: !!scrapedAttr
      });
      
      if (scrapedAttr) {
        scrapedMap.delete(goldenAttr.key.toLowerCase());
      }
    });

    let extraId = goldenAttrs.length;
    scrapedMap.forEach((attr) => {
      aligned.push({
        id: extraId++,
        key: attr.key,
        goldenValue: '',
        scrapedValue: attr.value,
        section: attr.section,
        isMatched: false,
        isExtra: true
      });
    });

    return aligned;
  };

  // Helper function to get description from golden product
  const getGoldenDescription = () => {
    if (!goldenProduct) return '';
    return goldenProduct.description || 
           (goldenProduct.features && goldenProduct.features.length > 0 ? goldenProduct.features.join(' • ') : '') ||
           'No description available';
  };

  // Helper function to get aligned attributes for both products
  const getAlignedAttributes = () => {
    if (!goldenProduct) return [];

    const goldenAttrs = [];
    const scrapedAttrs = [];
    
    // Get golden product attributes
    if (goldenProduct.details) {
      // MockData format
      Object.entries(goldenProduct.details).forEach(([key, value]) => {
        goldenAttrs.push({
          key: key.replace(/([A-Z])/g, ' $1').trim(),
          value: value,
          section: 'Product Details'
        });
      });
    } else {
      // Scraped format with sections
      if (goldenProduct.productDetailsArray) {
        goldenProduct.productDetailsArray.forEach(item => {
          goldenAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Product Details'
          });
        });
      }
      
      if (goldenProduct.manufacturingDetailsArray) {
        goldenProduct.manufacturingDetailsArray.forEach(item => {
          goldenAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Manufacturing Details'
          });
        });
      }
      
      if (goldenProduct.additionalInfoArray) {
        goldenProduct.additionalInfoArray.forEach(item => {
          goldenAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Additional Information'
          });
        });
      }
    }

    // Get scraped product attributes if available
    if (scrapedData) {
      if (scrapedData.productDetailsArray) {
        scrapedData.productDetailsArray.forEach(item => {
          scrapedAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Product Details'
          });
        });
      }
      
      if (scrapedData.manufacturingDetailsArray) {
        scrapedData.manufacturingDetailsArray.forEach(item => {
          scrapedAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Manufacturing Details'
          });
        });
      }
      
      if (scrapedData.additionalInfoArray) {
        scrapedData.additionalInfoArray.forEach(item => {
          scrapedAttrs.push({
            key: item.label || item.key || 'Attribute',
            value: item.value || 'N/A',
            section: 'Additional Information'
          });
        });
      }
    }

    // Create aligned list
    const aligned = [];
    const scrapedMap = new Map();
    
    // Map scraped attributes by key for quick lookup
    scrapedAttrs.forEach(attr => {
      scrapedMap.set(attr.key.toLowerCase(), attr);
    });

    // Add all golden attributes with matching scraped values
    goldenAttrs.forEach((goldenAttr, index) => {
      const scrapedAttr = scrapedMap.get(goldenAttr.key.toLowerCase());
      aligned.push({
        id: index,
        key: goldenAttr.key,
        goldenValue: goldenAttr.value,
        scrapedValue: scrapedAttr ? scrapedAttr.value : '',
        section: goldenAttr.section,
        isMatched: !!scrapedAttr
      });
      
      // Remove from map to track unmatched scraped attributes
      if (scrapedAttr) {
        scrapedMap.delete(goldenAttr.key.toLowerCase());
      }
    });

    // Add remaining scraped attributes that don't match golden product
    let extraId = goldenAttrs.length;
    scrapedMap.forEach((attr) => {
      aligned.push({
        id: extraId++,
        key: attr.key,
        goldenValue: '',
        scrapedValue: attr.value,
        section: attr.section,
        isMatched: false,
        isExtra: true
      });
    });

    return aligned;
  };

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
      // Prepare subcategory and product details from attributes
      const subcategory = 'Kurta Set'; // You can make this dynamic based on your product
      const productDetails = productAttributes.reduce((acc, attr) => {
        // Use attributeValues state which contains the current/edited values
        const value = attributeValues[attr.id];
        if (value) {
          acc[attr.key] = value;
        }
        return acc;
      }, {});

      console.log('AI Generate Title - Product Details:', productDetails);

      // Validate that we have product details
      if (Object.keys(productDetails).length === 0) {
        alert('Please fill in some product attributes before generating AI title.');
        setIsGeneratingTitle(false);
        return;
      }

      const generatedTitle = await generateProductTitle(subcategory, productDetails);
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
      // Prepare subcategory and product details from attributes
      const subcategory = 'Kurta Set'; // You can make this dynamic based on your product
      const productDetails = productAttributes.reduce((acc, attr) => {
        // Use attributeValues state which contains the current/edited values
        const value = attributeValues[attr.id];
        if (value) {
          acc[attr.key] = value;
        }
        return acc;
      }, {});

      console.log('AI Generate Description - Product Details:', productDetails);

      // Validate that we have product details
      if (Object.keys(productDetails).length === 0) {
        alert('Please fill in some product attributes before generating AI description.');
        setIsGeneratingDescription(false);
        return;
      }

      const generatedDescription = await generateProductDescription(subcategory, productDetails);
      setProductDescription(generatedDescription);
      setIsEditingDescription(true);
    } catch (error) {
      console.error('Error generating description:', error);
      alert(`Failed to generate description: ${error.message}`);
    } finally {
      setIsGeneratingDescription(false);
    }
  };

  // Attribute Handlers
  const handleEditAttribute = (attrId) => {
    setEditingAttributes(prev => ({ ...prev, [attrId]: true }));
  };

  const handleCancelEditAttribute = (attrId, originalValue) => {
    setAttributeValues(prev => ({ ...prev, [attrId]: originalValue }));
    setEditingAttributes(prev => ({ ...prev, [attrId]: false }));
  };

  const handleAcceptAttribute = (attrId) => {
    setProductAttributes(prev =>
      prev.map(attr =>
        attr.id === attrId ? { ...attr, accepted: true, scrapedValue: attributeValues[attrId] } : attr
      )
    );
    setEditingAttributes(prev => ({ ...prev, [attrId]: false }));
  };

  const handleAttributeChange = (attrId, value) => {
    setAttributeValues(prev => ({ ...prev, [attrId]: value }));
  };

  const handleAddAttribute = () => {
    const attributeKey = selectedAttributeType === 'custom' 
      ? newAttributeKey.trim().toLowerCase().replace(/\s+/g, '') 
      : selectedAttributeType;
    
    const attributeLabel = selectedAttributeType === 'custom' 
      ? newAttributeKey.trim() 
      : predefinedAttributes.find(a => a.key === selectedAttributeType)?.label || attributeKey;
    
    if (attributeKey && newAttributeValue.trim()) {
      const newId = productAttributes.length;
      const newAttr = {
        id: newId,
        key: attributeLabel,
        goldenValue: '',
        scrapedValue: newAttributeValue.trim(),
        accepted: false,
        isExtra: true
      };
      setProductAttributes([...productAttributes, newAttr]);
      setAttributeValues(prev => ({ ...prev, [newId]: newAttributeValue.trim() }));
      setNewAttributeKey('');
      setNewAttributeValue('');
      setSelectedAttributeType('');
      setShowAddAttribute(false);
    }
  };

  const handleProceed = () => {
    const allAccepted = productAttributes.every(attr => attr.accepted);
    if (allAccepted && scrapedData) {
      // Console log all the product data
      const productData = {
        title: acceptedTitle,
        description: acceptedDescription,
        attributes: productAttributes.map(attr => ({
          key: attr.key,
          value: attr.scrapedValue,
          accepted: attr.accepted
        }))
      };
      console.log('Product Data:', productData);
      
      // Save the existing/scraped product to localStorage for image comparison page
      const existingProduct = {
        id: scrapedData.asin || 'N/A',
        asin: scrapedData.asin || 'N/A',
        title: acceptedTitle,
        brand: scrapedData.brand,
        amazonUrl: productUrl,
        
        // Images from scraped data
        images: scrapedData.images || [],
        
        // Features/Bullets
        features: scrapedData.features || [],
        
        // Description
        description: acceptedDescription,
        
        // Detailed sections
        productDetails: scrapedData.productDetails,
        productDetailsArray: scrapedData.productDetailsArray || [],
        
        manufacturingDetails: scrapedData.manufacturingDetails,
        manufacturingDetailsArray: scrapedData.manufacturingDetailsArray || [],
        
        additionalInfo: scrapedData.additionalInfo,
        additionalInfoArray: scrapedData.additionalInfoArray || [],
        
        // Store the full scraped data
        scrapedData: scrapedData
      };
      
      localStorage.setItem('existingProduct', JSON.stringify(existingProduct));
      console.log('💾 Stored existing product in localStorage:', existingProduct);
      
      // Pass state to indicate NOT coming from new product page
      navigate('/compare-images', { state: { isNewProduct: false } });
    }
  };

  const renderStars = (rating) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;

    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(<Star key={i} className="h-4 w-4 fill-yellow-400 text-yellow-400" />);
      } else if (i === fullStars && hasHalfStar) {
        stars.push(<Star key={i} className="h-4 w-4 fill-yellow-400/50 text-yellow-400" />);
      } else {
        stars.push(<Star key={i} className="h-4 w-4 text-gray-300" />);
      }
    }
    return stars;
  };
  
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
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Product Comparison & Editing</h1>
          <p className="text-sm text-gray-600">
            Compare your product against the Gold Standard and enhance your listing with AI-powered suggestions
          </p>
        </div>

        {/* Main Content Container */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          
          {/* Product URL Input Section - Only show if no data loaded */}
          {!scrapedData && (
            <div className="mb-6 pb-6 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-700 mb-4">Load Product</h3>
              <div className="max-w-2xl">
                <div className="space-y-3">
                  <label className="block text-sm font-medium text-gray-700">
                    Enter Product URL
                  </label>
                  <input
                    type="text"
                    placeholder="https://www.amazon.in/dp/..."
                    className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    value={productUrl}
                    onChange={e => setProductUrl(e.target.value)}
                    disabled={isLoadingUrl}
                  />
                  <button
                    className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm w-full md:w-auto disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={handleLoadUrl}
                    disabled={isLoadingUrl || !productUrl.trim()}
                  >
                    {isLoadingUrl ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Loading...</span>
                      </>
                    ) : (
                      <>
                        <ExternalLink className="h-4 w-4" />
                        <span>Load URL</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}
          
          {/* Column Headers */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6 pb-4 border-b border-gray-200">
            {/* Left Column Header */}
            <div className="flex items-center gap-2">
              <Star className="h-6 w-6 fill-yellow-400 text-yellow-400" />
              <h2 className="text-xl font-semibold text-gray-900">Gold Standard Product</h2>
            </div>
            {/* Right Column Header */}
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Your Product</h2>
            </div>
          </div>

          {/* Product Attributes Section - MOVED TO TOP */}
          <div className="mb-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
              <h3 className="text-lg font-semibold text-gray-700">Product Attributes</h3>
              <h3 className="text-lg font-semibold text-gray-700">Product Attributes</h3>
            </div>
            
            {!scrapedData ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left: Gold Standard Attributes */}
                <div className="space-y-3">
                  {goldenProduct.details ? (
                    // MockData format
                    Object.entries(goldenProduct.details).map(([key, value], index) => (
                      <div key={index} className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-3">
                        <div className="text-xs font-semibold text-gray-600 mb-2 capitalize">
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </div>
                        <p className="text-sm font-medium text-gray-800">{value}</p>
                      </div>
                    ))
                  ) : (
                    // Scraped format with sections - DEDUPLICATED
                    (() => {
                      const allAttributes = [];
                      const seenKeys = new Set();
                      const normalizeKey = (key) => {
                        if (!key) return '';
                        // Remove ALL special characters, whitespace, newlines, colons, and convert to lowercase
                        return key.trim()
                          .replace(/[\n\r\s:‏‎]+/g, '') // Remove newlines, spaces, colons, and special Unicode chars
                          .toLowerCase();
                      };

                      // Collect all attributes from all sections with deduplication
                      if (goldenProduct.productDetailsArray && goldenProduct.productDetailsArray.length > 0) {
                        goldenProduct.productDetailsArray.forEach((item) => {
                          const key = item.label || item.key || 'Attribute';
                          const normalizedKey = normalizeKey(key);
                          if (!seenKeys.has(normalizedKey)) {
                            seenKeys.add(normalizedKey);
                            allAttributes.push(item);
                          }
                        });
                      }
                      
                      if (goldenProduct.manufacturingDetailsArray && goldenProduct.manufacturingDetailsArray.length > 0) {
                        goldenProduct.manufacturingDetailsArray.forEach((item) => {
                          const key = item.label || item.key || 'Attribute';
                          const normalizedKey = normalizeKey(key);
                          if (!seenKeys.has(normalizedKey)) {
                            seenKeys.add(normalizedKey);
                            allAttributes.push(item);
                          }
                        });
                      }
                      
                      if (goldenProduct.additionalInfoArray && goldenProduct.additionalInfoArray.length > 0) {
                        goldenProduct.additionalInfoArray.forEach((item) => {
                          const key = item.label || item.key || 'Attribute';
                          const normalizedKey = normalizeKey(key);
                          if (!seenKeys.has(normalizedKey)) {
                            seenKeys.add(normalizedKey);
                            allAttributes.push(item);
                          }
                        });
                      }

                      // Render deduplicated attributes
                      return allAttributes.map((item, index) => (
                        <div key={`attr-${index}`} className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-3">
                          <div className="text-xs font-semibold text-gray-600 mb-2">
                            {item.label || item.key || 'Attribute'}
                          </div>
                          <p className="text-sm font-medium text-gray-800">{item.value || 'N/A'}</p>
                        </div>
                      ));
                    })()
                  )}
                </div>

                {/* Right: Placeholder */}
                <div className="bg-gray-50 border border-gray-300 rounded-lg p-4 flex items-center justify-center">
                  <p className="text-gray-500 text-sm text-center">
                    Load a product URL to view attributes
                  </p>
                </div>
              </div>
            ) : (
              /* Aligned Attributes View */
              <div className="space-y-3">
                {productAttributes.map((attr) => (
                  <div key={attr.id} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Left: Golden Product Attribute */}
                    <div className={`rounded-lg p-3 ${
                      attr.goldenValue 
                        ? 'bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300' 
                        : 'bg-gray-100 border border-gray-300'
                    }`}>
                      <div className="text-xs font-semibold text-gray-600 mb-2">
                        {attr.key}
                      </div>
                      <p className="text-sm font-medium text-gray-800">
                        {attr.goldenValue || <span className="text-gray-400 italic">Not available</span>}
                      </p>
                    </div>

                    {/* Right: Scraped Product Attribute (Always Editable) */}
                    <div className="bg-white border border-gray-300 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-xs font-semibold text-gray-600">
                          {attr.key}
                          {attr.isExtra && (
                            <span className="ml-2 text-xs text-blue-600 font-normal">(Extra)</span>
                          )}
                        </div>
                        {attr.accepted && (
                          <div className="flex items-center text-green-600">
                            <Check className="h-4 w-4" />
                          </div>
                        )}
                      </div>

                      {editingAttributes[attr.id] ? (
                        <div className="space-y-2">
                          <input
                            type="text"
                            className="border border-gray-300 rounded-md px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            value={attributeValues[attr.id] || ''}
                            onChange={e => handleAttributeChange(attr.id, e.target.value)}
                            autoFocus
                          />
                          <div className="flex gap-2">
                            <button
                              className="flex items-center space-x-1 px-3 py-1.5 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
                              onClick={() => handleAcceptAttribute(attr.id)}
                            >
                              <Check className="h-3.5 w-3.5" />
                              <span>Accept</span>
                            </button>
                            <button
                              className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors text-sm"
                              onClick={() => handleCancelEditAttribute(attr.id, attr.scrapedValue)}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <p className="text-sm font-medium text-gray-800">
                            {attributeValues[attr.id] || <span className="text-gray-400 italic">Not available</span>}
                          </p>
                          <div className="flex gap-2 flex-shrink-0">
                            <button
                              className="flex items-center space-x-1 px-2 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-xs"
                              onClick={() => handleEditAttribute(attr.id)}
                            >
                              <Edit2 className="h-3 w-3" />
                              <span>Edit</span>
                            </button>
                            {!attr.accepted && (
                              <button
                                className="flex items-center space-x-1 px-2 py-1 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-xs"
                                onClick={() => handleAcceptAttribute(attr.id)}
                              >
                                <Check className="h-3 w-3" />
                                <span>Accept</span>
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
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
                      className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm"
                      onClick={() => setShowAddAttribute(true)}
                    >
                      <Plus className="h-4 w-4" />
                      <span>Add New Attribute</span>
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Product Title Section */}
          <div className="mb-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
              <h3 className="text-lg font-semibold text-gray-700">Product Title</h3>
              <h3 className="text-lg font-semibold text-gray-700">Product Title</h3>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Gold Standard Product Title */}
              <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-4">
                <div className="bg-white/60 rounded-md p-3 border border-yellow-200">
                  <p className="text-gray-800 text-sm leading-relaxed">
                    {goldenProduct.title}
                  </p>
                </div>
              </div>

              {/* Right: Your Product Title (Editable) or Placeholder */}
              {!scrapedData ? (
                <div className="bg-gray-50 border border-gray-300 rounded-lg p-4 flex items-center justify-center">
                  <p className="text-gray-500 text-sm text-center">
                    Load a product URL to view title
                  </p>
                </div>
              ) : (
                <div>
                  {isEditingTitle ? (
                    <div className="space-y-3">
                      <textarea
                        ref={titleTextareaRef}
                        className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                        value={productTitle}
                        onChange={e => setProductTitle(e.target.value)}
                        rows={3}
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
                        {acceptedTitle}
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <button
                          className="flex items-center space-x-1 px-3 py-1.5 bg-white border border-gray-300 rounded-md hover:bg-gray-100 transition-colors text-sm"
                          onClick={() => {
                            setProductTitle(originalTitle);
                            setIsEditingTitle(true);
                          }}
                        >
                          <Eye className="h-4 w-4" />
                          <span>View Original</span>
                        </button>
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
              )}
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
              <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-300 rounded-lg p-4">
                <div className="bg-white/60 rounded-md p-3 border border-yellow-200">
                  <p className="text-gray-800 text-sm leading-relaxed">
                    {getGoldenDescription()}
                  </p>
                </div>
              </div>

              {/* Right: Scraped Description or Placeholder */}
              {!scrapedData ? (
                <div className="bg-gray-50 border border-gray-300 rounded-lg p-4 flex items-center justify-center">
                  <p className="text-gray-500 text-sm text-center">
                    Load a product URL to view details
                  </p>
                </div>
              ) : (
                <div>
                  {isEditingDescription ? (
                    <div className="space-y-3">
                      <textarea
                        ref={descTextareaRef}
                        className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
                        value={productDescription}
                        onChange={e => setProductDescription(e.target.value)}
                        rows={4}
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
                        {acceptedDescription}
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <button
                          className="flex items-center space-x-1 px-3 py-1.5 bg-white border border-gray-300 rounded-md hover:bg-gray-100 transition-colors text-sm"
                          onClick={() => {
                            setProductDescription(originalDescription);
                            setIsEditingDescription(true);
                          }}
                        >
                          <Eye className="h-4 w-4" />
                          <span>View Original</span>
                        </button>
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
              )}
            </div>
          </div>

        </div>

        {/* Action Bar */}
        <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              {scrapedData ? (
                <>
                  {productAttributes.every(attr => attr.accepted) ? (
                    <>
                      <span className="font-medium text-green-600">All attributes accepted!</span>
                      <span className="ml-2">Ready to proceed to image comparison.</span>
                    </>
                  ) : (
                    <>
                      <span className="font-medium">Please accept all attributes to proceed</span>
                      <span className="ml-2">
                        ({productAttributes.filter(attr => attr.accepted).length} of {productAttributes.length} accepted)
                      </span>
                    </>
                  )}
                </>
              ) : (
                <>
                  <span className="font-medium">Enter a product URL to begin</span>
                  <span className="ml-2">Load a product to see the comparison</span>
                </>
              )}
            </div>
            <button
              className={`flex items-center space-x-2 px-6 py-2 rounded-md transition-colors text-sm font-medium ${
                scrapedData && productAttributes.every(attr => attr.accepted)
                  ? 'bg-green-600 text-white hover:bg-green-700 cursor-pointer'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
              onClick={handleProceed}
              disabled={!scrapedData || !productAttributes.every(attr => attr.accepted)}
            >
              <span>Proceed</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductComparator;
