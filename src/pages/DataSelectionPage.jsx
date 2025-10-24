// Page 2: Seller List and Data Selection Page

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchSheetData, fetchSellerData } from '../api/api';
import { ChevronDown, Download, ArrowRight, Loader2, Search, X } from 'lucide-react';

const DataSelectionPage = () => {
  const [selectedSeller, setSelectedSeller] = useState('');
  const [sellerData, setSellerData] = useState(null);
  const [rawFilteredData, setRawFilteredData] = useState(null);
  const [tableHeaders, setTableHeaders] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingDropdown, setIsLoadingDropdown] = useState(true);
  const [error, setError] = useState(null);
  const [availableOptions, setAvailableOptions] = useState([]);
  const [rawSheetData, setRawSheetData] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const navigate = useNavigate();

  // Fetch available sellers and stores on component mount
  useEffect(() => {
    const loadAvailableSellers = async () => {
      setIsLoadingDropdown(true);
      try {
        const sheetData = await fetchSheetData();
        setRawSheetData(sheetData.data);
        
        // Store headers from API
        if (sheetData.headers && sheetData.headers.length > 0) {
          setTableHeaders(sheetData.headers);
        }
        
        // Create combined options with store names and seller IDs
        const storeMap = new Map();
        
        sheetData.data.forEach(item => {
          const sellerId = item.seller_id || item.Seller_ID || item['Seller ID'];
          const storeName = item.store_name || item.Store_Name || item['Store Name'] || item.store || item.Store;
          
          if (sellerId && storeName) {
            const key = `${storeName}-${sellerId}`;
            if (!storeMap.has(key)) {
              storeMap.set(key, {
                id: sellerId,
                storeName: storeName,
                displayName: `${storeName} - (${sellerId})`
              });
            }
          }
        });
        
        const combinedOptions = Array.from(storeMap.values());
        setAvailableOptions(combinedOptions);
      } catch (err) {
        console.error('Failed to load sellers:', err);
        setAvailableOptions([]);
      } finally {
        setIsLoadingDropdown(false);
      }
    };
    
    loadAvailableSellers();
  }, []);

  // Filter options based on search term
  const filteredOptions = availableOptions.filter(option =>
    option.displayName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    option.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    option.storeName.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSellerChange = async (sellerId, storeName) => {
    if (!sellerId) {
      setSelectedSeller('');
      setSellerData(null);
      setRawFilteredData(null);
      setError(null);
      setIsDropdownOpen(false);
      setSearchTerm('');
      return;
    }

    setSelectedSeller(sellerId);
    setIsLoading(true);
    setError(null);
    setSellerData(null);
    setRawFilteredData(null);
    setIsDropdownOpen(false);
    setSearchTerm('');

    try {
      // Filter the raw sheet data by seller_id
      const sheetData = await fetchSheetData();
      const filteredRawData = sheetData.data.filter(item => {
        const itemSellerId = item.seller_id || item.Seller_ID || item['Seller ID'];
        return itemSellerId === sellerId;
      });
      
      setRawFilteredData(filteredRawData);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProcess = () => {
    if (selectedSeller) {
      navigate(`/setup/${selectedSeller}`);
    }
  };

  const handleDownload = () => {
    if (!rawFilteredData || rawFilteredData.length === 0) {
      alert('No data available to download');
      return;
    }

    // Use dynamic headers from API
    const headers = tableHeaders.length > 0 ? tableHeaders : ['Category', 'Gender', 'Age Group', 'Subcategory', 'URL'];
    
    // Download the exact raw data retrieved from API
    const csvContent = [
      headers.join(','),
      ...rawFilteredData.map(item => 
        headers.map(header => `"${item[header] || ''}"`).join(',')
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    const todayDate = new Date().toISOString().split('T')[0];
    a.href = url;
    a.download = `${selectedSeller}_${todayDate}_raw_data.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Clear search and close dropdown
  const clearSearch = () => {
    setSearchTerm('');
  };

  // Get selected store name for display
  const getSelectedDisplayName = () => {
    if (!selectedSeller) return '';
    const selectedOption = availableOptions.find(option => option.id === selectedSeller);
    return selectedOption ? selectedOption.displayName : selectedSeller;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="space-y-8">
          {/* Header */}
          <div className="text-center space-y-3">
            <h1 className="text-2xl font-bold text-gray-900">Select Seller Data</h1>
            <p className="text-base text-gray-600 max-w-2xl mx-auto">
              Search and select by Store Name or Seller ID to load listing data and begin the optimization process.
            </p>
          </div>

          {/* Seller Selection */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="max-w-2xl mx-auto">
              <label htmlFor="seller-select" className="block text-sm font-medium text-gray-700 mb-2">
                Search Store or Seller
              </label>
              
              {/* Custom Dropdown */}
              <div className="relative">
                {/* Dropdown Trigger */}
                <button
                  onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                  className="w-full px-3 py-2 bg-white border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-left text-sm flex items-center justify-between cursor-pointer hover:bg-gray-50"
                  disabled={isLoading || isLoadingDropdown}
                >
                  <span className="truncate">
                    {selectedSeller ? getSelectedDisplayName() : 'Select a store or seller...'}
                  </span>
                  {isLoadingDropdown ? (
                    <Loader2 className="h-4 w-4 text-blue-600 animate-spin flex-shrink-0" />
                  ) : (
                    <ChevronDown className={`h-4 w-4 text-gray-400 flex-shrink-0 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
                  )}
                </button>

                {/* Dropdown Menu */}
                {isDropdownOpen && (
                  <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-80 overflow-hidden">
                    {/* Search Input */}
                    <div className="p-2 border-b border-gray-200">
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                        <input
                          type="text"
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          placeholder="Search by store name or seller ID..."
                          className="w-full pl-10 pr-8 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                          autoFocus
                        />
                        {searchTerm && (
                          <button
                            onClick={clearSearch}
                            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                          >
                            <X className="h-4 w-4" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Options List */}
                    <div className="overflow-y-auto max-h-60">
                      {filteredOptions.length === 0 ? (
                        <div className="px-4 py-3 text-sm text-gray-500 text-center">
                          {isLoadingDropdown ? 'Loading...' : 'No stores or sellers found'}
                        </div>
                      ) : (
                        filteredOptions.map((option) => (
                          <button
                            key={`${option.storeName}-${option.id}`}
                            onClick={() => handleSellerChange(option.id, option.storeName)}
                            className={`w-full px-4 py-2 text-left text-sm hover:bg-gray-100 focus:bg-gray-100 focus:outline-none ${
                              selectedSeller === option.id ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
                            }`}
                          >
                            <div className="font-medium">{option.storeName}</div>
                            <div className="text-xs text-gray-500 mt-1">Seller ID: {option.id}</div>
                          </button>
                        ))
                      )}
                    </div>

                    {/* Results Count */}
                    {filteredOptions.length > 0 && (
                      <div className="px-4 py-2 border-t border-gray-200 bg-gray-50 text-xs text-gray-500">
                        {filteredOptions.length} result{filteredOptions.length !== 1 ? 's' : ''} found
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Selected Info */}
              {selectedSeller && (
                <div className="mt-3 p-3 bg-blue-50 rounded-md">
                  <div className="text-sm text-blue-700">
                    <span className="font-medium">Selected: </span>
                    {getSelectedDisplayName()}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
              <div className="flex flex-col items-center justify-center space-y-4">
                <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
                <div className="text-center">
                  <h3 className="text-lg font-medium text-gray-900">Loading Product Data</h3>
                  <p className="text-gray-600">Fetching data for {getSelectedDisplayName()}...</p>
                </div>
                {/* Skeleton loader */}
                <div className="w-full max-w-4xl space-y-3 mt-6">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="animate-pulse flex space-x-4 p-4 bg-gray-50 rounded">
                      <div className="h-4 bg-gray-300 rounded w-20"></div>
                      <div className="h-4 bg-gray-300 rounded flex-1"></div>
                      <div className="h-4 bg-gray-300 rounded w-16"></div>
                      <div className="h-4 bg-gray-300 rounded w-20"></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <div className="text-center">
                <h3 className="text-lg font-medium text-red-900">Error Loading Data</h3>
                <p className="text-red-700 mt-2">{error}</p>
                <button
                  onClick={() => handleSellerChange(selectedSeller)}
                  className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}

          {/* Data Loaded State */}
          {rawFilteredData && !isLoading && (
            <div 
              className="bg-white rounded-lg shadow-sm border border-gray-200 opacity-0 translate-y-4 animate-fadeInUp"
              style={{
                animation: 'fadeInUp 0.6s ease-out forwards'
              }}
            >
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-base font-semibold text-gray-900">
                  Listing Data for {getSelectedDisplayName()}
                </h2>
                <p className="text-gray-600 mt-1 text-m">
                  Listings Found: <span className="font-bold text-blue-600">{rawFilteredData.length}</span>
                </p>
              </div>

              {/* Data Table */}
              <div className="overflow-y-auto max-h-[500px]">
                <table className="w-full">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      {tableHeaders.length > 0 ? (
                        tableHeaders.map((header, index) => (
                          <th key={index} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            {header}
                          </th>
                        ))
                      ) : (
                        <>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Store Name
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Category
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Gender
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Age Group
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Subcategory
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            URL
                          </th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {rawFilteredData.map((item, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        {tableHeaders.length > 0 ? (
                          tableHeaders.map((header, colIndex) => (
                            <td key={colIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                              {header === 'URL' && item[header] ? (
                                <a 
                                  href={item[header]} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:text-blue-800 underline truncate block max-w-xs"
                                >
                                  View Product
                                </a>
                              ) : (
                                <span className={colIndex === 0 ? 'text-gray-900' : ''}>
                                  {item[header] || '-'}
                                </span>
                              )}
                            </td>
                          ))
                        ) : (
                          <>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {item.store_name || item.Store_Name || item['Store Name'] || '-'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                              {item.Category || '-'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                              {item.Gender || '-'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                              {item['Age Group'] || '-'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                              {item.Subcategory || '-'}
                            </td>
                            <td className="px-6 py-4 text-sm text-gray-600">
                              {item.URL ? (
                                <a 
                                  href={item.URL} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:text-blue-800 underline truncate block max-w-xs"
                                >
                                  View Product
                                </a>
                              ) : (
                                <span className="text-gray-400">No URL</span>
                              )}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Action Buttons */}
              <div className="p-4 bg-gray-50 border-t border-gray-200">
                <div className="flex items-center justify-center space-x-3">
                  <button
                    onClick={handleDownload}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
                  >
                    <Download className="h-4 w-4" />
                    <span className="font-medium">Download Sheet</span>
                  </button>
                  <button
                    onClick={handleProcess}
                    className="flex items-center space-x-2 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm"
                  >
                    <span className="font-medium">Process</span>
                    <ArrowRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* How it Works Section */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="text-center mb-6">
              <h2 className="text-xl font-bold text-gray-900 mb-2">How It Works</h2>
              <p className="text-gray-600 text-sm">Follow these simple steps to optimize your Amazon product listings</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-lg font-bold">
                  1
                </div>
                <h3 className="font-semibold text-gray-900 mb-2 text-sm">Select Data</h3>
                <p className="text-gray-600 text-xs">Search and select by Store Name or Seller ID to load your listing data</p>
              </div>
              
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="w-12 h-12 bg-green-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-lg font-bold">
                  2
                </div>
                <h3 className="font-semibold text-gray-900 mb-2 text-sm">Compare & Setup</h3>
                <p className="text-gray-600 text-xs">View the gold standard and decide whether to edit existing or create new listings</p>
              </div>
              
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="w-12 h-12 bg-purple-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-lg font-bold">
                  3
                </div>
                <h3 className="font-semibold text-gray-900 mb-2 text-sm">AI Optimization</h3>
                <p className="text-gray-600 text-xs">Use AI tools to enhance titles, descriptions, and generate better product listing images</p>
              </div>
              
              <div className="text-center p-4 bg-orange-50 rounded-lg">
                <div className="w-12 h-12 bg-orange-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-lg font-bold">
                  4
                </div>
                <h3 className="font-semibold text-gray-900 mb-2 text-sm">Publish & Track</h3>
                <p className="text-gray-600 text-xs">Review final changes and publish your optimized listings to Amazon</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Close dropdown when clicking outside */}
      {isDropdownOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setIsDropdownOpen(false)}
        />
      )}

      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(16px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
};

export default DataSelectionPage;