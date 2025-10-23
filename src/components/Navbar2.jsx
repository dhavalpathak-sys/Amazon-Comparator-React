import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { Package, LogOut, HelpCircle, Info } from 'lucide-react';

const Navbar = () => {
  const { sellerId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Show seller ID on pages that have it in the URL
  const showSellerId = sellerId && location.pathname !== '/';

  const handleHomeClick = () => {
    navigate('/');
  };

  return (
    <nav className="bg-gray-900 border-b border-orange-400 px-4 py-2">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 cursor-pointer" onClick={handleHomeClick}>
            <Package className="h-6 w-6 text-orange-400" />
            <h1 className="text-lg font-bold text-white hover:text-orange-400 transition-colors">
              Listro
            </h1>
          </div>
          
          {showSellerId && (
            <>
              <div className="h-4 w-px bg-gray-600"></div>
              <div className="flex items-center space-x-3">
                <span className="text-xs font-medium text-gray-300">Seller ID:</span>
                <span className="px-2 py-1 bg-orange-500 text-white rounded text-xs font-semibold">
                  {sellerId}
                </span>
                {/* Placeholder block as specified */}
                <div className="w-16 h-6 bg-gray-700 rounded"></div>
              </div>
            </>
          )}
        </div>

        {/* Right side buttons */}
        <div className="flex items-center space-x-2">
          <button className="flex items-center space-x-1 px-3 py-1 text-gray-300 hover:text-white text-xs rounded hover:bg-gray-800 transition-colors">
            <HelpCircle className="h-4 w-4" />
            <span>Help</span>
          </button>
          <button className="flex items-center space-x-1 px-3 py-1 text-gray-300 hover:text-white text-xs rounded hover:bg-gray-800 transition-colors">
            <Info className="h-4 w-4" />
            <span>About</span>
          </button>
          <button className="flex items-center space-x-1 px-3 py-1 hover:text-white text-xs rounded hover:bg-gray-800 transition-colors border border-gray-700 bg-orange-400">
            <LogOut className="h-4 w-4" />
            <span>Logout</span>
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;