import React from 'react';
import Sidebar from './Sidebar';
import { Search, Bell, User } from 'lucide-react';

const Layout = ({ children }) => {
    return (
        <div className="min-h-screen bg-slate-50 font-['Inter']">
            <Sidebar />

            {/* Main Content Wrapper */}
            <div className="ml-64 min-h-screen flex flex-col">

                {/* Top Header */}
                <header className="h-20 bg-white/80 backdrop-blur-md border-b border-slate-100 sticky top-0 z-40 px-8 flex items-center justify-between">

                    {/* Search Bar */}
                    <div className="relative w-96">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
                        <input
                            type="text"
                            placeholder="Search students, records..."
                            className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-500 transition-all text-sm"
                        />
                    </div>

                    {/* Right Actions */}
                    <div className="flex items-center gap-6">
                        <button className="relative p-2 text-slate-400 hover:text-slate-600 transition-colors">
                            <Bell size={20} />
                            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full border-2 border-white"></span>
                        </button>

                        <div className="flex items-center gap-3 pl-6 border-l border-slate-100">
                            <div className="text-right hidden md:block">
                                <p className="text-sm font-semibold text-slate-900">Admin User</p>
                                <p className="text-xs text-slate-500">Principal</p>
                            </div>
                            <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center border-2 border-white shadow-sm text-indigo-600 font-bold">
                                AU
                            </div>
                        </div>
                    </div>
                </header>

                {/* Page Content */}
                <main className="flex-1 p-8">
                    {children}
                </main>
            </div>
        </div>
    );
};

export default Layout;
