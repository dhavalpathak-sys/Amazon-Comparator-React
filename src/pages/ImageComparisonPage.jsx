// Page 5: Image Comparison and AI Generation Page

import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { generateAIImage } from '../api/api';
import { Wand2, Loader2, ArrowRight, X, Upload, Download, Link, Images } from 'lucide-react';

const ImageComparisonPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [previewImage, setPreviewImage] = useState(null);
  const [referenceImage, setReferenceImage] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [productAttributes, setProductAttributes] = useState({});
  const [goldStandardProduct, setGoldStandardProduct] = useState(null);
  const [existingProduct, setExistingProduct] = useState(null);
  // Get isNewProduct from navigation state
  const [isNewProduct, setIsNewProduct] = useState(location.state?.isNewProduct || false);
  
  // New state for existing images dialog
  const [showExistingImagesDialog, setShowExistingImagesDialog] = useState(false);

  // Load product data from localStorage on component mount
  useEffect(() => {
    // Check if coming from new product page via navigation state
    if (location.state?.isNewProduct) {
      console.log('✅ Coming from new product page - hiding existing product section');
    }

    // Load product attributes/filters
    const storedFilters = localStorage.getItem('productFilters');
    if (storedFilters) {
      try {
        const filters = JSON.parse(storedFilters);
        setProductAttributes(filters);
        console.log('✅ Loaded product attributes from localStorage:', filters);
      } catch (error) {
        console.error('Error parsing stored filters:', error);
      }
    }

    // Load gold standard product
    const storedGoldStandard = localStorage.getItem('goldStandardProduct');
    if (storedGoldStandard) {
      try {
        const goldProduct = JSON.parse(storedGoldStandard);
        setGoldStandardProduct(goldProduct);
        console.log('✅ Loaded gold standard product from localStorage:', goldProduct);
      } catch (error) {
        console.error('Error parsing stored gold standard product:', error);
      }
    }

    // Load existing product only if not coming from new product page
    if (!location.state?.isNewProduct) {
      const storedExisting = localStorage.getItem('existingProduct');
      if (storedExisting) {
        try {
          const existing = JSON.parse(storedExisting);
          setExistingProduct(existing);
          console.log('✅ Loaded existing product from localStorage:', existing);
        } catch (error) {
          console.error('Error parsing stored existing product:', error);
        }
      }
    }
  }, [location.state]);

  // Get images from loaded products (up to 7 images each)
  const getGoldStandardImages = () => {
    const emptyArray = Array(7).fill(null);
    
    if (goldStandardProduct && goldStandardProduct.images && goldStandardProduct.images.length > 0) {
      // Filter out duplicates by using Set with image URLs
      const uniqueImages = [...new Set(goldStandardProduct.images)];
      const images = uniqueImages.slice(0, 7);
      
      console.log('📸 Gold Standard - Total images:', goldStandardProduct.images.length, 'Unique images:', uniqueImages.length);
      
      // Pad with null if less than 7
      while (images.length < 7) {
        images.push(null);
      }
      return images;
    }
    
    console.log('⚠️ No gold standard images found, returning empty array');
    return emptyArray; // Return empty array if no images
  };

  const getExistingImages = () => {
    const emptyArray = Array(7).fill(null);
    
    if (existingProduct && existingProduct.images && existingProduct.images.length > 0) {
      // Filter out duplicates by using Set with image URLs
      const uniqueImages = [...new Set(existingProduct.images)];
      const images = uniqueImages.slice(0, 7);
      
      console.log('📸 Existing Product - Total images:', existingProduct.images.length, 'Unique images:', uniqueImages.length);
      
      // Pad with null if less than 7
      while (images.length < 7) {
        images.push(null);
      }
      return images;
    }
    
    console.log('⚠️ No existing product images found, returning empty array');
    return emptyArray; // Return empty array if no images
  };

  // Get all unique existing images for the dialog
  const getAllExistingImages = () => {
    if (existingProduct && existingProduct.images && existingProduct.images.length > 0) {
      // Filter out duplicates and return all images
      const uniqueImages = [...new Set(existingProduct.images)];
      return uniqueImages;
    }
    return [];
  };

  // Function to convert image URL to File object
  const urlToFile = async (imageUrl, filename) => {
    try {
      console.log('🔄 Converting URL to File:', imageUrl);
      
      // Fetch the image
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch image: ${response.status}`);
      }
      
      // Get the image as blob
      const blob = await response.blob();
      
      // Create a File object from the blob
      const file = new File([blob], filename, { 
        type: blob.type || 'image/jpeg',
        lastModified: Date.now()
      });
      
      console.log('✅ Successfully converted URL to File:', file);
      return file;
    } catch (error) {
      console.error('❌ Error converting URL to File:', error);
      throw error;
    }
  };

  const [generatedImages, setGeneratedImages] = useState(
    Array(7).fill(null).map(() => ({ url: null, isLoading: false }))
  );

  // Track which image is selected per row (0 = existing, 1 = generated)
  const [selectedImages, setSelectedImages] = useState(Array(7).fill(null));

  const imageLabels = [
    'Main Product',
    'Front View',
    'Back View', 
    'Side View',
    'Detail Shot 1',
    'Detail Shot 2',
    'Lifestyle'
  ];

  const handleGenerateImage = async (index) => {
    if (!referenceFile && !referenceImage) {
      alert('Please upload or select a reference image first!');
      return;
    }

    // Check if style index is not yet implemented on backend (indices 5 and 6)
    if (index >= 5) {
      alert('🚧 This image style is coming soon! Currently in development. Stay tuned! 🎨');
      return;
    }

    const imageType = imageLabels[index];
    // Use index as style_index (0-6 for the 7 image slots)
    const styleIndex = index;
    
    // Normalize attribute keys to match prompt template placeholders
    // Convert filter keys like "Subcategory" -> "SubCategory", "Age Group" -> "AgeGroup"
    const normalizedAttributes = {};
    Object.entries(productAttributes).forEach(([key, value]) => {
      // Handle specific key mappings
      if (key === 'Subcategory') {
        normalizedAttributes['SubCategory'] = value;
      } else if (key === 'Age Group') {
        normalizedAttributes['AgeGroup'] = value;
      } else {
        // Keep other keys as-is (Gender, Category, etc.)
        normalizedAttributes[key] = value;
      }
    });
    
    console.log('🎨 Generating image with attributes:', normalizedAttributes);
    
    setGeneratedImages(prev => 
      prev.map((img, i) => 
        i === index ? { ...img, isLoading: true } : img
      )
    );

    try {
      let imageFile = referenceFile;
      
      // If we have a referenceImage URL but no file, convert the URL to a File
      if (!referenceFile && referenceImage) {
        console.log('🔄 Converting reference image URL to File...');
        try {
          imageFile = await urlToFile(referenceImage, `reference-image-${Date.now()}.jpg`);
          console.log('✅ Successfully converted reference image to File');
        } catch (conversionError) {
          console.error('❌ Failed to convert reference image:', conversionError);
          throw new Error('Failed to process reference image. Please try uploading the image directly instead.');
        }
      }

      if (!imageFile) {
        throw new Error('No reference image available for generation');
      }

      console.log('🚀 Calling generateAIImage with file:', imageFile);
      
      // Call the real API with reference image file, style index, and attributes
      const newImageUrl = await generateAIImage(imageFile, styleIndex, normalizedAttributes);
      setGeneratedImages(prev => 
        prev.map((img, i) => 
          i === index ? { url: newImageUrl, isLoading: false } : img
        )
      );
    } catch (error) {
      console.error('Error generating image:', error);
      
      // Provide more helpful error messages
      let errorMessage = error.message;
      if (errorMessage.includes('No image returned from model')) {
        errorMessage = 'The AI model could not generate this image. This might be due to content safety filters or prompt issues. Please try a different style or image.';
      } else if (errorMessage.includes('style_index out of range')) {
        errorMessage = 'This image style is not yet available. Please try another style.';
      } else if (errorMessage.includes('safety filters')) {
        errorMessage = 'Content generation was blocked by safety filters. Please try a different reference image or style.';
      } else if (errorMessage.includes('No image uploaded')) {
        errorMessage = 'Reference image processing failed. Please try uploading the image directly or select a different existing image.';
      } else if (errorMessage.includes('Failed to process reference image')) {
        errorMessage = error.message; // Use the specific error message from conversion
      }
      
      alert(`Failed to generate image: ${errorMessage}`);
      setGeneratedImages(prev => 
        prev.map((img, i) => 
          i === index ? { ...img, isLoading: false } : img
        )
      );
    }
  };

  const handleReferenceImageUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setReferenceFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setReferenceImage(e.target.result);
      };
      reader.readAsDataURL(file);
      console.log('✅ Reference image uploaded:', file.name);
    }
  };

  const handleSelectExistingImage = async (imageUrl) => {
    try {
      console.log('🔄 Selecting existing image as reference:', imageUrl);
      
      // Convert the selected image URL to a File object
      const file = await urlToFile(imageUrl, `existing-reference-${Date.now()}.jpg`);
      
      // Set both the file and the image URL for display
      setReferenceFile(file);
      setReferenceImage(imageUrl);
      setShowExistingImagesDialog(false);
      
      console.log('✅ Successfully set existing image as reference');
    } catch (error) {
      console.error('❌ Failed to select existing image:', error);
      alert('Failed to select this image as reference. Please try uploading the image directly or select a different image.');
    }
  };

  const handleRemoveReferenceImage = () => {
    setReferenceImage(null);
    setReferenceFile(null);
    console.log('🗑️ Reference image removed');
  };

  const handleDownloadImage = (imageUrl, imageName) => {
    try {
      // Create a temporary link element and trigger download
      const link = document.createElement('a');
      link.href = imageUrl;
      link.download = `${imageName.replace(/\s+/g, '_')}.png`;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error downloading image:', error);
      alert('Failed to download image. Please try again.');
    }
  };

  const handleCopyImageLink = (imageUrl) => {
    navigator.clipboard.writeText(imageUrl).then(() => {
      alert('Image link copied to clipboard!');
    }).catch(err => {
      console.error('Failed to copy link:', err);
      alert('Failed to copy link. Please try again.');
    });
  };

  const handleImageSelection = (rowIndex, imageType) => {
    // imageType: 0 = existing, 1 = generated
    const newSelection = [...selectedImages];
    
    // Toggle selection: if clicking on already selected, deselect it
    if (newSelection[rowIndex] === imageType) {
      newSelection[rowIndex] = null;
    } else {
      newSelection[rowIndex] = imageType;
    }
    
    setSelectedImages(newSelection);
    console.log(`🖱️ Row ${rowIndex} selection changed to:`, imageType === 0 ? 'existing' : 'generated');
  };

  const handleSaveChanges = () => {
    // For new products: require all 7 images to be generated
    if (isNewProduct) {
      const generatedCount = generatedImages.filter(img => img.url !== null).length;
      if (generatedCount < 7) {
        alert(`⚠️ Please generate all 7 images before proceeding. Currently generated: ${generatedCount}/7`);
        return;
      }
    } else {
      // For existing products: require exactly 7 selections (one per row)
      const selectedCount = selectedImages.filter(sel => sel !== null).length;
      if (selectedCount < 7) {
        alert(`⚠️ Please select one image per row (7 total). Currently selected: ${selectedCount}/7`);
        return;
      }
    }
    
    const allImages = [];
    const generatedImageUrls = [];
    
    if (isNewProduct) {
      // For new products: collect all generated images
      generatedImages.forEach((imgData, index) => {
        if (imgData.url) {
          allImages.push({
            url: imgData.url,
            type: 'generated',
            label: imageLabels[index] || `Generated ${index + 1}`
          });
          generatedImageUrls.push(imgData.url);
        }
      });
    } else {
      // For existing products: collect only selected images
      const existingImages = getExistingImages();
      
      selectedImages.forEach((selection, index) => {
        if (selection === 0) {
          // User selected existing image
          const existingUrl = existingImages[index];
          if (existingUrl) {
            allImages.push({
              url: existingUrl,
              type: 'existing',
              label: imageLabels[index] || `Image ${index + 1}`
            });
          }
        } else if (selection === 1) {
          // User selected generated image
          const generatedUrl = generatedImages[index].url;
          if (generatedUrl) {
            allImages.push({
              url: generatedUrl,
              type: 'generated',
              label: imageLabels[index] || `Generated ${index + 1}`
            });
            generatedImageUrls.push(generatedUrl);
          }
        }
      });
    }
    
    console.log('📦 Selected/Generated images:', allImages.length);
    console.log('🎨 Generated images only:', generatedImageUrls.length);
    console.log('📋 All images data:', allImages);
    
    // Save both to localStorage
    localStorage.setItem('generatedImages', JSON.stringify(generatedImageUrls));
    localStorage.setItem('allProductImages', JSON.stringify(allImages));
    
    // Navigate to final page
    navigate('/final-page');
  };

  const ImageCard = ({ src, alt, label, type, index = null, onGenerate = null, isLoading = false, isSelected = false, onSelect = null }) => (
    <div className={`bg-white rounded-md border-2 overflow-hidden mb-2 transition-all ${
      isSelected ? 'border-green-500 shadow-lg ring-2 ring-green-300' : 'border-gray-200'
    } ${onSelect && src ? 'cursor-pointer hover:border-blue-300' : ''}`}
    onClick={() => onSelect && src && !isLoading && onSelect()}
    >
      <div className="aspect-square bg-gray-50 flex items-center justify-center relative">
        {/* Selection indicator */}
        {isSelected && (
          <div className="absolute top-1 right-1 bg-green-500 text-white rounded-full p-1 z-20 shadow-lg">
            <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
            </svg>
          </div>
        )}
        
        {/* For reference/current types with no image */}
        {(type === 'reference' || type === 'current') && !src && (
          <div className="flex flex-col items-center justify-center p-3 text-center">
            <div className="text-gray-400 text-xs">No image</div>
          </div>
        )}
        
        {/* For generate type - show generate button when no image */}
        {type === 'generate' && !src && !isLoading && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onGenerate(index);
            }}
            className="flex flex-col items-center justify-center p-3 text-center hover:bg-gray-100 transition-colors w-full h-full"
          >
            <Wand2 className="h-5 w-5 text-blue-600 mb-1" />
            <span className="text-xs font-medium text-gray-700">Generate</span>
          </button>
        )}
        
        {/* Loading state */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center p-3 text-center">
            <Loader2 className="h-5 w-5 text-blue-600 animate-spin mb-1" />
            <span className="text-xs font-medium text-gray-700">Generating...</span>
          </div>
        )}
        
        {/* Show image when available */}
        {src && !isLoading && (
          <>
            <img
              src={src}
              alt={alt}
              className="w-full h-full object-contain hover:opacity-90 transition-opacity"
              onClick={(e) => {
                e.stopPropagation();
                setPreviewImage({ src, alt, label });
              }}
              onError={(e) => {
                console.error('Image load error for:', src);
                e.target.src = `https://placehold.co/250x250/E5E7EB/6B7280?text=${encodeURIComponent(alt)}`;
              }}
            />
            {type === 'generate' && (
              <div className="absolute bottom-1 left-1 right-1 flex gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onGenerate(index);
                  }}
                  className="flex-1 bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700 transition-colors z-10"
                  title="Regenerate Image"
                >
                  Regenerate
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownloadImage(src, label);
                  }}
                  className="bg-green-600 text-white p-1 rounded text-xs hover:bg-green-700 transition-colors z-10"
                  title="Download Image"
                >
                  <Download className="h-3 w-3" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCopyImageLink(src);
                  }}
                  className="bg-purple-600 text-white p-1 rounded text-xs hover:bg-purple-700 transition-colors z-10"
                  title="Copy Image Link"
                >
                  <Link className="h-3 w-3" />
                </button>
              </div>
            )}
          </>
        )}
      </div>
      <div className="p-2">
        <h4 className="text-xs font-medium text-gray-900 text-center">{label}</h4>
        {onSelect && src && !isLoading && (
          <p className="text-xs text-center text-gray-500 mt-1">
            {isSelected ? '✓ Selected' : 'Click to select'}
          </p>
        )}
      </div>
    </div>
  );

  // Existing Images Dialog Component
  const ExistingImagesDialog = () => {
    const existingImages = getAllExistingImages();
    
    return (
      <div 
        className="fixed inset-0 flex items-center justify-center z-50 p-4 bg-black bg-opacity-50"
        onClick={() => setShowExistingImagesDialog(false)}
      >
        <div 
          className="relative bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[85vh] overflow-hidden border-4 border-blue-500 animate-scaleIn"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Select Existing Image as Reference</h3>
              <p className="text-sm text-gray-600 mt-1">
                Choose an image from your existing product images to use as reference for AI generation
              </p>
            </div>
            <button
              onClick={() => setShowExistingImagesDialog(false)}
              className="text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full p-1 transition-all duration-200"
            >
              <X className="h-6 w-6" />
            </button>
          </div>
          
          <div className="p-6 bg-gray-50 max-h-[65vh] overflow-y-auto">
            {existingImages.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {existingImages.map((imageUrl, index) => (
                  <div
                    key={index}
                    className="bg-white rounded-lg border-2 border-gray-200 overflow-hidden cursor-pointer hover:border-blue-400 hover:shadow-md transition-all"
                    onClick={() => handleSelectExistingImage(imageUrl)}
                  >
                    <div className="aspect-square bg-gray-100 flex items-center justify-center">
                      <img
                        src={imageUrl}
                        alt={`Existing product image ${index + 1}`}
                        className="w-full h-full object-contain"
                        onError={(e) => {
                          e.target.src = `https://placehold.co/200x200/E5E7EB/6B7280?text=Image+${index + 1}`;
                        }}
                      />
                    </div>
                    <div className="p-2 text-center">
                      <span className="text-xs font-medium text-gray-700">Image {index + 1}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Images className="h-12 w-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-600">No existing product images found.</p>
                <p className="text-sm text-gray-500 mt-1">Please upload images in the previous steps.</p>
              </div>
            )}
          </div>
          
          <div className="flex items-center justify-between p-4 border-t border-gray-200 bg-gray-50">
            <span className="text-sm text-gray-600">
              {existingImages.length} image(s) available
            </span>
            <button
              onClick={() => setShowExistingImagesDialog(false)}
              className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Preview Modal */}
      {previewImage && (
        <div 
          className="fixed inset-0 flex items-center justify-center z-50 p-4"
          onClick={() => setPreviewImage(null)}
        >
          <div 
            className="relative bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[85vh] overflow-hidden border-4 border-blue-500 animate-scaleIn"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50">
              <h3 className="text-lg font-semibold text-gray-900">{previewImage.label}</h3>
              <button
                onClick={() => setPreviewImage(null)}
                className="text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full p-1 transition-all duration-200"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            <div className="p-6 bg-gray-50">
              <img
                src={previewImage.src}
                alt={previewImage.alt}
                className="w-full h-auto max-h-[65vh] object-contain rounded-lg shadow-md"
                onError={(e) => {
                  e.target.src = `https://placehold.co/800x800/E5E7EB/6B7280?text=${encodeURIComponent(previewImage.alt)}`;
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Existing Images Dialog */}
      {showExistingImagesDialog && <ExistingImagesDialog />}

      <div className="max-w-7xl mx-auto px-4 py-4">
        {/* Header */}
        <div className="text-center mb-4">
          <h1 className="text-xl font-bold text-gray-900 mb-1">Image Comparison & Generation</h1>
          <p className="text-sm text-gray-600">
            Compare your current images against the Gold Standard and generate new AI-powered alternatives for each slot.
          </p>
          {Object.keys(productAttributes).length > 0 && (
            <div className="mt-2 inline-flex items-center space-x-2 bg-blue-50 border border-blue-200 px-3 py-1 rounded-full text-xs">
              <span className="font-medium text-blue-900">Using attributes:</span>
              {Object.entries(productAttributes).map(([key, value]) => (
                <span key={key} className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                  {key}: {value}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Reference Image Upload Section */}
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg shadow-md border-2 border-purple-300 p-4 mb-4">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <h3 className="text-base font-bold text-gray-900 mb-1">Reference Image for AI Generation</h3>
              <p className="text-xs text-gray-600">
                {referenceImage 
                  ? 'Reference image selected. You can now generate AI images.' 
                  : 'Upload a reference image or select from existing images to guide the AI generation process'
                }
              </p>
              {referenceImage && (
                <div className="mt-2 flex items-center space-x-2 text-xs">
                  <div className="bg-green-100 text-green-800 px-2 py-1 rounded flex items-center">
                    <svg className="h-3 w-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                    Ready for AI Generation
                  </div>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-3">
              {!referenceImage ? (
                <>
                  <label className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 cursor-pointer transition-colors text-sm">
                    <Upload className="h-4 w-4" />
                    <span>Upload Reference</span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleReferenceImageUpload}
                      className="hidden"
                    />
                  </label>
                  {!isNewProduct && getAllExistingImages().length > 0 && (
                    <button
                      onClick={() => setShowExistingImagesDialog(true)}
                      className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
                    >
                      <Images className="h-4 w-4" />
                      <span>Select Existing Image</span>
                    </button>
                  )}
                </>
              ) : (
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <img
                      src={referenceImage}
                      alt="Reference"
                      className="w-20 h-20 object-cover rounded-lg border-2 border-purple-400 cursor-pointer hover:opacity-80 transition-opacity"
                      onClick={() => setPreviewImage({ src: referenceImage, alt: 'Reference Image', label: 'Reference Image' })}
                    />
                    <button
                      onClick={handleRemoveReferenceImage}
                      className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="flex flex-col space-y-2">
                    <label className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 cursor-pointer transition-colors text-sm">
                      <Upload className="h-4 w-4" />
                      <span>Change</span>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleReferenceImageUpload}
                        className="hidden"
                      />
                    </label>
                    {!isNewProduct && getAllExistingImages().length > 0 && (
                      <button
                        onClick={() => setShowExistingImagesDialog(true)}
                        className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
                      >
                        <Images className="h-4 w-4" />
                        <span>Select Different</span>
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Two or Three Column Layout with Overall Border */}
        <div className="bg-white rounded-lg shadow-lg border-2 border-gray-300 p-4">
          {isNewProduct ? (
            // New Product: Two columns layout (Gold Standard + AI Generated)
            <div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
                <p className="text-sm text-blue-900 text-center font-medium">
                  ⚠️ For new products, please generate all 7 images before proceeding
                </p>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Gold Standard Images */}
                <div className="bg-yellow-50 border-2 border-yellow-300 rounded-lg p-3">
                  <div className="text-center mb-3">
                    <h2 className="text-base font-bold text-gray-900 mb-1">Gold Standard Images</h2>
                    <p className="text-xs text-gray-600">Reference benchmark images</p>
                  </div>
                  <div className="space-y-1">
                    {getGoldStandardImages().map((src, index) => (
                      <ImageCard
                        key={`gold-${index}`}
                        src={src}
                        alt={`Gold standard ${imageLabels[index]}`}
                        label={imageLabels[index]}
                        type="reference"
                      />
                    ))}
                  </div>
                </div>

                {/* AI Generated Images */}
                <div className="border-2 border-blue-300 rounded-lg p-3">
                  <div className="text-center mb-3">
                    <h2 className="text-base font-bold text-gray-900 mb-1">AI Generated Images</h2>
                    <p className="text-xs text-gray-600">Create optimized alternatives</p>
                    <div className="mt-2 inline-block bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-xs font-medium">
                      {generatedImages.filter(img => img.url !== null).length} / 7 Generated
                    </div>
                  </div>
                  <div className="space-y-1">
                    {generatedImages.map((imgData, index) => (
                      <ImageCard
                        key={`generated-${index}`}
                        src={imgData.url}
                        alt={`AI generated ${imageLabels[index]}`}
                        label={imageLabels[index]}
                        type="generate"
                        index={index}
                        onGenerate={handleGenerateImage}
                        isLoading={imgData.isLoading}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Existing Product: Row-based selection layout
            <div>
              <div className="space-y-4">
                {imageLabels.map((label, index) => {
                  const existingImg = getExistingImages()[index];
                  const generatedImg = generatedImages[index];
                  
                  return (
                    <div key={`row-${index}`} className="bg-gray-50 border-2 border-gray-300 rounded-lg p-3">
                      <div className="text-center mb-2">
                        <h3 className="text-sm font-bold text-gray-900">{label}</h3>
                        <p className="text-xs text-gray-600">Select one image for this slot</p>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Gold Standard (reference only) */}
                        <div>
                          <div className="text-center mb-2">
                            <span className="text-xs font-semibold text-yellow-700 bg-yellow-100 px-2 py-1 rounded">
                              Gold Standard
                            </span>
                          </div>
                          <ImageCard
                            src={getGoldStandardImages()[index]}
                            alt={`Gold standard ${label}`}
                            label="Reference"
                            type="reference"
                          />
                        </div>
                        
                        {/* Existing Image (selectable) */}
                        <div>
                          <div className="text-center mb-2">
                            <span className="text-xs font-semibold text-gray-700 bg-gray-200 px-2 py-1 rounded">
                              Your Existing
                            </span>
                          </div>
                          <ImageCard
                            src={existingImg}
                            alt={`Existing ${label}`}
                            type="current"
                            isSelected={selectedImages[index] === 0}
                            onSelect={existingImg ? () => handleImageSelection(index, 0) : null}
                          />
                        </div>
                        
                        {/* Generated Image (selectable) */}
                        <div>
                          <div className="text-center mb-2">
                            <span className="text-xs font-semibold text-blue-700 bg-blue-100 px-2 py-1 rounded">
                              AI Generated
                            </span>
                          </div>
                          <ImageCard
                            src={generatedImg.url}
                            alt={`AI generated ${label}`}
                            type="generate"
                            index={index}
                            onGenerate={handleGenerateImage}
                            isLoading={generatedImg.isLoading}
                            isSelected={selectedImages[index] === 1}
                            onSelect={generatedImg.url ? () => handleImageSelection(index, 1) : null}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Action Bar */}
        <div className="mt-4 bg-white rounded-lg shadow-sm border border-gray-200 p-3">
          <div className="flex items-center justify-between">
            <div className="text-xs text-gray-600">
              {isNewProduct ? (
                <>
                  <span className="font-medium">Generated Images:</span>
                  <span className="ml-2">
                    {generatedImages.filter(img => img.url !== null).length} of {generatedImages.length} completed
                  </span>
                </>
              ) : (
                <>
                  <span className="font-medium">Selected Images:</span>
                  <span className="ml-2">
                    {selectedImages.filter(sel => sel !== null).length} of 7 selected
                  </span>
                </>
              )}
            </div>
            <button
              onClick={handleSaveChanges}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors text-sm ${
                (isNewProduct && generatedImages.filter(img => img.url !== null).length === 7) ||
                (!isNewProduct && selectedImages.filter(sel => sel !== null).length === 7)
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
              disabled={
                isNewProduct 
                  ? generatedImages.filter(img => img.url !== null).length < 7
                  : selectedImages.filter(sel => sel !== null).length < 7
              }
            >
              <ArrowRight className="h-4 w-4" />
              <span>Proceed</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageComparisonPage;