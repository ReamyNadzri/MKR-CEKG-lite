const KuihListing = () => {
    const kuihClasses = [
        'Kuih Keria',
        'Kuih Ketayap',
        'Kuih Lapis',
        'Kuih Seri Muka',
        'Onde Onde',
        'Kuih Talam',
        'Kuih Cara',
        'Tepung Pelita',
        'Kuih Bingka',
        'Kuih Bahulu',
        'Apam Balik',
    ];

    return (
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <h3 className="font-semibold text-slate-700 mb-4 text-sm uppercase tracking-wider">
                Identifiable Kuih
            </h3>
            <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto">
                {kuihClasses.map((kuih, index) => (
                    <span
                        key={index}
                        className="px-3 py-1 bg-slate-100 text-slate-600 text-xs rounded-full border border-slate-200"
                    >
                        {kuih}
                    </span>
                ))}
            </div>
        </div>
    );
};

export default KuihListing;
