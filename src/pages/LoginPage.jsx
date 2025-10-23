import { useNavigate, useLocation } from "react-router-dom";
import { GoogleLogin } from "@react-oauth/google";
import { jwtDecode } from "jwt-decode";

const LoginPage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    
    return (
        <div>
            <div className="w-full h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-orange-50">
                <div className="bg-white rounded-2xl shadow-lg p-10 w-110 h-96">
                    <div className="text-center mb-6">
                        <div className="inline-flex items-center mb-5">
                            <div className="ml-2 text-orange-600 text-3xl font-bold">Listro</div>
                        </div>
                        <h2 className="text-3xl font-bold text-gray-900 p-2">Welcome to Listro</h2>
                        <div className="mt-4 flex items-center justify-center">
                            <p className="text-gray-600 mt-2">Please sign in to continue.</p>
                        </div>
                    </div>
                    {/* <button className="w-full flex items-center justify-center bg-white border border-gray-300 
                    rounded-xl shadow-sm py-3 hover:shadow-md transition-shadow duration-200" onClick={() => navigate('/seller-list')}>
                        <img className="pr-5" src="/src/assets/Google_logo.png" alt="Logo"/>
                        <span className="text-gray-700 text-lg font-medium">Sign in with Google</span>
                    </button> */}
                    <div className="flex justify-center items-center">
                        <GoogleLogin
                            onSuccess={credentialResponse => {
                                try {
                                    console.log('Google login response:', credentialResponse);
                                    const decoded = jwtDecode(credentialResponse.credential);
                                    console.log('Decoded token:', decoded);
                                    
                                    // Store user information in localStorage
                                    localStorage.setItem('user', JSON.stringify({
                                        name: decoded.name,
                                        email: decoded.email,
                                        picture: decoded.picture,
                                        token: credentialResponse.credential
                                    }));

                                    console.log('User logged in as:', decoded.name);
                                    navigate('/seller-list');
                                } catch (error) {
                                    console.error('Error during login:', error);
                                    alert('Login failed. Please try again.');
                                }
                            }}
                            onError={(error) => {
                                console.error('Login Failed:', error);
                                localStorage.removeItem('user');
                                alert('Login failed. Please try again.');
                            }}
                            logo_alignment="left"
                            type="standard"
                            theme="outline"
                            size="large"
                            useOneTap
                        />
                    </div>
                    <p className="text-center text-blue-500 text-sm mt-5">Only Vaco Binary emails can login</p>
                </div>
            </div>
        </div>
    );
}

export default LoginPage;