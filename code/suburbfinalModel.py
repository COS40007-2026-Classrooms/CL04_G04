#  CL04_G04 Group Project
#  VIC FUEL PRICE FORECAST  --  SUBURB LEVEL  (per postcode)
#  Extension of fuel_price_forecast.py
#
#  ARCHITECTURE:
#    State forecast (from fuel_price_forecast.py)
#        +
#    Postcode spread (calibrated from live API station data)
#        =
#    Suburb forecast per postcode
#
#  SPREAD MODEL (3 tiers, in priority order):
#    Tier 1 -- Live API:   postcode has 1+ stations in current API response
#              -> use median of actual station prices as this week's anchor
#              -> spread = postcode_median - state_median
#
#    Tier 2 -- Historical: postcode has no live stations but appeared in
#              past API calls (spread stored in calibration CSV)
#              -> use stored historical median spread
#
#    Tier 3 -- Zone model: postcode never seen in API data
#              -> use zone-based spread estimate:
#                 inner_metro:  base = -1.5 cpl  (competition-dense, low spread)
#                 middle_metro: base = +2.0 cpl
#                 regional_vic: base = +8.0 cpl  (transport cost)
#                 rural_remote: base = +18.0 cpl
#
#  USAGE:
#    python suburb_price_forecast.py
#    python suburb_price_forecast.py --api-key YOUR_KEY
#
#  INPUTS:
#    models/ulp91_price_model.h5   -- trained state-level Ridge model
#    models/ulp95_price_model.h5
#    models/diesel_price_model.h5
#    whole_fleet_*_by_postcode_*.csv  -- VicRoads fleet data (postcode list)
#
#  OUTPUTS:
#    reports/suburb_price_forecast.csv   -- one row per postcode
#    reports/suburb_price_forecast.json  -- web app payload
#    reports/suburb_price_report.txt     -- human readable
#    data/postcode_calibration.csv       -- running spread history (auto-updated)
# ============================

import os, sys, warnings, json, re, uuid, argparse
warnings.filterwarnings('ignore')

import pandas as pd
pd.options.future.infer_string = False
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request, urllib.error

#  CONFIG
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = SCRIPT_DIR.parent
MDL_DIR    = ROOT_DIR / 'models'
RPT_DIR    = ROOT_DIR / 'reports';  RPT_DIR.mkdir(exist_ok=True)
DAT_DIR    = ROOT_DIR / 'data';     DAT_DIR.mkdir(exist_ok=True)

# Calibration file -- stores running postcode spread history
CALIB_FILE = DAT_DIR / 'postcode_calibration.csv'

# Fair Fuel Open Data API
API_BASE   = 'https://api.fuel.service.vic.gov.au/open-data/v1'

# Excise cut
EXCISE_CUT_START = datetime(2026, 4, 1)
EXCISE_CUT_END   = datetime(2026, 6, 30)
EXCISE_CUT_CPL   = 32.0

# Zone spread parameters (base cpl above state median, from ACCC regional data)
ZONE_SPREADS = {
    'inner_metro':  {'base': +2.0,  'std': 4.0},   # Melbourne CBD / inner suburbs
    'middle_metro': {'base': +4.0,  'std': 5.0},   # outer Melbourne ring
    'regional_vic': {'base': +10.0, 'std': 6.0},   # Geelong, Ballarat, Bendigo etc
    'rural_remote': {'base': +20.0, 'std': 8.0},   # far regional
}

# Fuel type codes: API code -> our internal name
FUEL_CODES = {'U91': 'ulp91', 'P95': 'ulp95', 'DSL': 'diesel'}


#  POSTCODE -> SUBURB NAME LOOKUP
#  Complete hardcoded map for all 679 VIC postcodes in the fleet dataset
#  Source: Australia Post official locality data + ABS
#  For postcodes covering multiple suburbs, the primary/best-known name is used

POSTCODE_SUBURBS = {
    # Non-VIC border/special
    '2020': 'Mascot (NSW)', '2640': 'Albury (NSW)', '2641': 'Albury South (NSW)',
    '2731': 'Moama (NSW)', '9000': 'International Mail',
    # Inner Metro 3000-3207
    '3000': 'Melbourne CBD', '3002': 'East Melbourne', '3003': 'West Melbourne',
    '3004': 'St Kilda Road', '3006': 'Southbank', '3008': 'Docklands',
    '3011': 'Footscray', '3012': 'Brooklyn', '3013': 'Yarraville',
    '3015': 'Newport', '3016': 'Williamstown', '3018': 'Altona',
    '3019': 'Altona North', '3020': 'Sunshine', '3021': 'Sunshine North',
    '3022': 'Albanvale', '3023': 'Deer Park', '3024': 'Werribee South',
    '3025': 'Laverton', '3026': 'Laverton North', '3027': 'Williams Landing',
    '3028': 'Hoppers Crossing', '3029': 'Hoppers Crossing', '3030': 'Point Cook',
    '3031': 'Flemington', '3032': 'Essendon', '3033': 'Keilor East',
    '3034': 'Keilor', '3036': 'Keilor Downs', '3037': 'Delahey',
    '3038': 'Taylors Lakes', '3039': 'Moonee Ponds', '3040': 'Pascoe Vale South',
    '3041': 'Pascoe Vale', '3042': 'Airport West', '3043': 'Tullamarine',
    '3044': 'Glenroy', '3045': 'Melbourne Airport', '3046': 'Glenroy North',
    '3047': 'Dallas', '3048': 'Broadmeadows', '3049': 'Meadow Heights',
    '3050': 'Royal Melbourne Hospital', '3051': 'North Melbourne', '3052': 'Parkville',
    '3053': 'Carlton', '3054': 'Carlton North', '3055': 'Brunswick',
    '3056': 'Brunswick East', '3057': 'Brunswick East', '3058': 'Coburg',
    '3059': 'Coburg North', '3060': 'Fawkner', '3061': 'Campbellfield',
    '3062': 'Somerton', '3063': 'Craigieburn', '3064': 'Craigieburn',
    '3065': 'Fitzroy', '3066': 'Collingwood', '3067': 'Abbotsford',
    '3068': 'Clifton Hill', '3070': 'Northcote', '3071': 'Thornbury',
    '3072': 'Preston', '3073': 'Reservoir', '3074': 'Reservoir',
    '3075': 'Thomastown', '3076': 'Lalor', '3078': 'Alphington',
    '3079': 'Ivanhoe', '3081': 'Heidelberg', '3082': 'Mill Park',
    '3083': 'Bundoora', '3084': 'Greensborough', '3085': 'Macleod',
    '3087': 'Watsonia', '3088': 'Montmorency', '3089': 'Eltham',
    '3090': 'Eltham North', '3091': 'Yarrambat', '3093': 'Lower Plenty',
    '3094': 'Templestowe', '3095': 'Hurstbridge', '3096': 'Diamond Creek',
    '3097': 'Kangaroo Ground', '3099': 'Panton Hill', '3101': 'Kew',
    '3102': 'Kew East', '3103': 'Balwyn', '3104': 'Balwyn North',
    '3105': 'Bulleen', '3106': 'Doncaster', '3107': 'Doncaster East',
    '3108': 'Deepdene', '3109': 'Box Hill North', '3111': 'Box Hill',
    '3113': 'Warrandyte', '3114': 'Park Orchards', '3115': 'Wonga Park',
    '3116': 'Chirnside Park', '3121': 'Richmond', '3122': 'Hawthorn',
    '3123': 'Hawthorn East', '3124': 'Camberwell', '3125': 'Box Hill South',
    '3126': 'Canterbury', '3127': 'Surrey Hills', '3128': 'Mont Albert',
    '3129': 'Mont Albert North', '3130': 'Nunawading', '3131': 'Mitcham',
    '3132': 'Ringwood', '3133': 'Ringwood East', '3134': 'Ringwood North',
    '3135': 'Bayswater', '3136': 'Croydon', '3137': 'Kilsyth',
    '3138': 'Lilydale', '3139': 'Yarra Glen', '3140': 'Yering',
    '3141': 'South Yarra', '3142': 'Toorak', '3143': 'Armadale',
    '3144': 'Malvern', '3145': 'Malvern East', '3146': 'Glen Iris',
    '3147': 'Ashburton', '3148': 'Chadstone', '3149': 'Mount Waverley',
    '3150': 'Glen Waverley', '3151': 'Burwood', '3152': 'Knox City',
    '3153': 'Bayswater North', '3154': 'The Basin', '3155': 'Ferntree Gully',
    '3156': 'Boronia', '3158': 'Upwey', '3159': 'Tecoma',
    '3160': 'Belgrave', '3161': 'Caulfield', '3162': 'Caulfield South',
    '3163': 'Moorabbin', '3164': 'Narre Warren East', '3165': 'Oakleigh South',
    '3166': 'Oakleigh', '3167': 'Oakleigh East', '3168': 'Clayton',
    '3169': 'Clayton South', '3170': 'Mulgrave', '3171': 'Springvale',
    '3172': 'Springvale South', '3173': 'Noble Park', '3174': 'Noble Park North',
    '3175': 'Dandenong', '3176': 'Dandenong South', '3177': 'Endeavour Hills',
    '3178': 'Rowville', '3179': 'Scoresby', '3180': 'Knoxfield',
    '3181': 'Prahran', '3182': 'St Kilda', '3183': 'Balaclava',
    '3184': 'Brighton East', '3185': 'Elwood', '3186': 'Brighton',
    '3187': 'Brighton East', '3188': 'Hampton', '3189': 'Moorabbin',
    '3190': 'Highett', '3191': 'Sandringham', '3192': 'Cheltenham',
    '3193': 'Black Rock', '3194': 'Mentone', '3195': 'Mordialloc',
    '3196': 'Aspendale', '3197': 'Patterson Lakes', '3198': 'Frankston South',
    '3199': 'Frankston', '3200': 'Frankston North', '3201': 'Carrum Downs',
    '3202': 'Seaford', '3204': 'Moorabbin', '3205': 'South Melbourne',
    '3206': 'Albert Park', '3207': 'Port Melbourne',
    # Outer Metro / Geelong belt 3209-3502
    '3209': 'Altona Meadows', '3210': 'Laverton North', '3211': 'Little River',
    '3212': 'Lara', '3213': 'Corio', '3214': 'Norlane',
    '3215': 'North Geelong', '3216': 'Geelong', '3217': 'Grovedale',
    '3218': 'Belmont', '3219': 'Hamlyn Heights', '3220': 'Geelong West',
    '3221': 'Highton', '3222': 'Ocean Grove', '3223': 'Queenscliff',
    '3224': 'Leopold', '3225': 'Portarlington', '3226': 'Drysdale',
    '3227': 'Barwon Heads', '3228': 'Torquay', '3230': 'Aireys Inlet',
    '3231': 'Anglesea', '3232': 'Lorne', '3233': 'Apollo Bay',
    '3234': 'Lorne', '3235': 'Colac East', '3236': 'Lavers Hill',
    '3238': 'Princetown', '3239': 'Birregurra', '3240': 'Winchelsea',
    '3241': 'Inverleigh', '3242': 'Bannockburn', '3243': 'Teesdale',
    '3249': 'Cobden', '3250': 'Camperdown', '3251': 'Noorat',
    '3254': 'Camperdown', '3260': 'Mortlake', '3264': 'Timboon',
    '3265': 'Warrnambool', '3266': 'Timboon', '3267': 'Port Fairy',
    '3268': 'Port Fairy', '3269': 'Koroit', '3270': 'Macarthur',
    '3271': 'Hamilton', '3272': 'Coleraine', '3273': 'Casterton',
    '3274': 'Hamilton', '3275': 'Dartmoor', '3276': 'Macarthur',
    '3277': 'Portland', '3278': 'Heywood', '3279': 'Koroit',
    '3280': 'Warrnambool', '3281': 'Allansford', '3282': 'Terang',
    '3283': 'Panmure', '3284': 'Peterborough', '3285': 'Port Campbell',
    '3289': 'Dunkeld', '3291': 'Hawkesdale', '3292': 'Penshurst',
    '3293': 'Dunkeld', '3294': 'Balmoral', '3300': 'Hamilton',
    '3301': 'Coleraine', '3302': 'Casterton', '3303': 'Merino',
    '3304': 'Dartmoor', '3305': 'Portland', '3307': 'Heywood',
    '3310': 'Macarthur', '3311': 'Cavendish', '3312': 'Dunkeld',
    '3315': 'Balmoral', '3317': 'Edenhope', '3318': 'Kaniva',
    '3321': 'Bannockburn', '3322': 'Meredith', '3324': 'Rokewood',
    '3325': 'Lismore', '3328': 'Inverleigh', '3330': 'Lethbridge',
    '3331': 'Meredith', '3335': 'Melton West', '3336': 'Melton',
    '3337': 'Melton South', '3338': 'Melton', '3340': 'Bacchus Marsh',
    '3341': 'Ballan', '3342': 'Greendale', '3345': 'Trentham',
    '3350': 'Ballarat', '3351': 'Ballarat East', '3352': 'Ballarat North',
    '3353': 'Wendouree', '3354': 'Mount Clear', '3355': 'Delacombe',
    '3356': 'Canadian', '3357': 'Miners Rest', '3358': 'Invermay Park',
    '3360': 'Smythesdale', '3361': 'Skipton', '3363': 'Clunes',
    '3364': 'Creswick', '3370': 'Maryborough', '3371': 'Dunolly',
    '3373': 'Ararat', '3374': 'Elmhurst', '3375': 'Lexton',
    '3377': 'Stawell', '3378': 'Glenorchy', '3379': 'Ararat',
    '3380': 'Horsham', '3381': 'Natimuk', '3382': 'Dimboola',
    '3384': 'Murtoa', '3385': 'Rupanyup', '3387': 'Warracknabeal',
    '3388': 'Hopetoun', '3390': 'Jeparit', '3391': 'Rainbow',
    '3393': 'Sea Lake', '3395': 'Ouyen', '3400': 'Horsham',
    '3401': 'Horsham', '3407': 'Natimuk', '3409': 'Nhill',
    '3412': 'Dimboola', '3413': 'Warracknabeal', '3418': 'Ouyen',
    '3420': 'Sea Lake', '3423': 'Pyramid Hill', '3424': 'Wedderburn',
    '3427': 'Gisborne', '3428': 'Riddells Creek', '3429': 'Sunbury',
    '3430': 'Clarkefield', '3431': 'Romsey', '3432': 'Lancefield',
    '3433': 'Macedon', '3434': 'Kyneton', '3437': 'Woodend',
    '3440': 'Gisborne South', '3441': 'New Gisborne', '3442': 'Riddells Creek',
    '3444': 'Kyneton', '3446': 'Newstead', '3447': 'Castlemaine',
    '3450': 'Castlemaine', '3453': 'Harcourt', '3458': 'Daylesford',
    '3460': 'Creswick', '3461': 'Clunes', '3462': 'Talbot',
    '3464': 'Maryborough', '3465': 'Maryborough', '3467': 'Avoca',
    '3472': 'Dunolly', '3475': 'St Arnaud', '3477': 'Charlton',
    '3478': 'Donald', '3480': 'Wycheproof', '3482': 'Birchip',
    '3483': 'Charlton', '3487': 'Kerang', '3488': 'Cohuna',
    '3489': 'Gunbower', '3490': 'Pyramid Hill', '3491': 'Rochester',
    '3494': 'Echuca', '3496': 'Swan Hill', '3500': 'Mildura',
    '3501': 'Mildura', '3502': 'Mildura',
    # Regional VIC 3505-3999
    '3505': 'Merbein', '3506': 'Red Cliffs', '3512': 'Charlton',
    '3515': 'Bendigo', '3516': 'Bendigo', '3518': 'Wedderburn',
    '3521': 'Rochester', '3525': 'Swan Hill', '3529': 'Kerang',
    '3530': 'Cohuna', '3537': 'Echuca', '3540': 'Echuca',
    '3542': 'Rochester', '3544': 'Kyabram', '3549': 'Rushworth',
    '3550': 'Bendigo', '3551': 'Bendigo', '3554': 'Golden Square',
    '3555': 'Kangaroo Flat', '3556': 'Eaglehawk', '3558': 'Epsom',
    '3559': 'Elmore', '3563': 'Cohuna', '3564': 'Echuca',
    '3565': 'Rochester', '3568': 'Kyabram', '3570': 'Castlemaine',
    '3572': 'Maldon', '3575': 'Heathcote', '3576': 'Rushworth',
    '3579': 'Shepparton', '3580': 'Shepparton', '3581': 'Mooroopna',
    '3583': 'Tatura', '3584': 'Tongala', '3585': 'Cobram',
    '3586': 'Yarrawonga', '3588': 'Numurkah', '3589': 'Nathalia',
    '3591': 'Kyabram', '3594': 'Echuca', '3595': 'Swan Hill',
    '3596': 'Robinvale', '3599': 'Kerang', '3607': 'Seymour',
    '3608': 'Nagambie', '3610': 'Euroa', '3612': 'Violet Town',
    '3614': 'Nagambie', '3616': 'Shepparton', '3617': 'Tatura',
    '3618': 'Tongala', '3620': 'Benalla', '3621': 'Benalla',
    '3623': 'Mansfield', '3629': 'Wangaratta', '3630': 'Shepparton',
    '3631': 'Shepparton', '3632': 'Shepparton', '3633': 'Shepparton',
    '3634': 'Shepparton', '3635': 'Cobram', '3636': 'Nathalia',
    '3637': 'Numurkah', '3638': 'Yarrawonga', '3639': 'Cobram',
    '3640': 'Cobram', '3641': 'Yarrawonga', '3643': 'Yarrawonga',
    '3644': 'Yarrawonga', '3646': 'Wangaratta', '3647': 'Wangaratta',
    '3649': 'Bright', '3658': 'Broadford', '3659': 'Broadford',
    '3660': 'Seymour', '3663': 'Broadford', '3664': 'Broadford',
    '3666': 'Avenel', '3669': 'Euroa', '3670': 'Benalla',
    '3671': 'Benalla', '3672': 'Benalla', '3675': 'Wangaratta',
    '3676': 'Wangaratta', '3677': 'Wangaratta', '3678': 'Wangaratta',
    '3683': 'Glenrowan', '3685': 'Wodonga', '3687': 'Wodonga',
    '3688': 'Wodonga', '3689': 'Wodonga', '3690': 'Wodonga',
    '3691': 'Wodonga', '3694': 'Rutherglen', '3695': 'Corryong',
    '3698': 'Mount Beauty', '3699': 'Bright', '3700': 'Corryong',
    '3704': 'Omeo', '3707': 'Orbost', '3711': 'Alexandra',
    '3712': 'Yea', '3713': 'Yea', '3714': 'Eildon',
    '3715': 'Mansfield', '3717': 'Mansfield', '3718': 'Mansfield',
    '3719': 'Kinglake', '3720': 'Healesville', '3722': 'Marysville',
    '3723': 'Buxton', '3724': 'Alexandra', '3725': 'Moyhu',
    '3727': 'Myrtleford', '3730': 'Bright', '3732': 'Porepunkah',
    '3735': 'Buckland', '3737': 'Mount Beauty', '3740': 'Beechworth',
    '3741': 'Yackandandah', '3744': 'Chiltern', '3746': 'Rutherglen',
    '3747': 'Beechworth', '3749': 'Wangaratta', '3750': 'South Morang',
    '3751': 'Doreen', '3752': 'Mernda', '3753': 'Whittlesea',
    '3754': 'Doreen', '3755': 'Mernda', '3756': 'Wallan',
    '3757': 'Beveridge', '3758': 'Kilmore', '3760': 'Broadford',
    '3761': 'Wallan', '3762': 'Kilmore', '3764': 'Kilmore',
    '3765': 'Montrose', '3766': 'Mooroolbark', '3767': 'Kilsyth South',
    '3770': 'Yering', '3775': 'Kangaroo Ground', '3777': 'Hurstbridge',
    '3779': 'Kinglake West', '3781': 'Gembrook', '3782': 'Cockatoo',
    '3783': 'Emerald', '3785': 'Ferntree Gully', '3786': 'Olinda',
    '3788': 'Selby', '3791': 'Sassafras', '3792': 'Belgrave Heights',
    '3793': 'Belgrave South', '3795': 'Launching Place', '3797': 'Warburton',
    '3800': 'Monash', '3802': 'Endeavour Hills', '3803': 'Hallam',
    '3804': 'Narre Warren', '3805': 'Narre Warren South', '3806': 'Berwick',
    '3807': 'Harkaway', '3810': 'Pakenham', '3812': 'Beaconsfield',
    '3813': 'Nar Nar Goon', '3815': 'Bunyip', '3816': 'Garfield',
    '3818': 'Drouin', '3820': 'Warragul', '3821': 'Yarragon',
    '3822': 'Trafalgar', '3825': 'Moe', '3831': 'Traralgon',
    '3835': 'Morwell', '3840': 'Traralgon', '3842': 'Morwell',
    '3844': 'Traralgon', '3847': 'Rosedale', '3850': 'Sale',
    '3854': 'Heyfield', '3856': 'Maffra', '3857': 'Stratford',
    '3858': 'Bairnsdale', '3859': 'Heyfield', '3860': 'Bairnsdale',
    '3869': 'Mirboo North', '3870': 'Boolarra', '3871': 'Yinnar',
    '3873': 'Foster', '3875': 'Bairnsdale', '3878': 'Orbost',
    '3880': 'Orbost', '3882': 'Marlo', '3885': 'Cann River',
    '3887': 'Cann River', '3888': 'Mallacoota', '3889': 'Cann River',
    '3891': 'Mallacoota', '3892': 'Delegate', '3895': 'Orbost',
    '3896': 'Swifts Creek', '3898': 'Omeo', '3900': 'Metung',
    '3902': 'Eagle Point', '3903': 'Paynesville', '3904': 'Metung',
    '3910': 'Langwarrin', '3911': 'Baxter', '3912': 'Somerville',
    '3913': 'Tyabb', '3914': 'Hastings', '3915': 'Pearcedale',
    '3916': 'Moorooduc', '3919': 'Tyabb', '3920': 'Bittern',
    '3922': 'Cowes', '3925': 'San Remo', '3926': 'Dromana',
    '3927': 'Red Hill', '3930': 'Frankston', '3933': 'Pearcedale',
    '3934': 'Somerville', '3936': 'Hastings', '3938': 'Baxter',
    '3940': 'Bittern', '3942': 'Flinders', '3945': 'Mornington',
    '3946': 'Mount Eliza', '3950': 'Leongatha', '3951': 'Korumburra',
    '3953': 'Mirboo North', '3954': 'Leongatha', '3956': 'Inverloch',
    '3957': 'Wonthaggi', '3960': 'Meeniyan', '3964': 'Foster',
    '3965': 'Welshpool', '3966': 'Toora', '3967': 'Yarram',
    '3971': 'Yarram', '3975': 'Cranbourne', '3976': 'Cranbourne South',
    '3977': 'Cranbourne South', '3978': 'Cranbourne West', '3979': 'Tooradin',
    '3980': 'Koo Wee Rup', '3981': 'Pakenham', '3984': 'Longwarry',
    '3987': 'Bunyip', '3988': 'Garfield', '3990': 'Drouin West',
    '3991': 'Drouin', '3992': 'Warragul', '3995': 'Corinella',
    '3996': 'Wonthaggi',
}

# Additional postcodes (smaller localities)
POSTCODE_SUBURBS.update({
    '3237': 'Beech Forest',
    '3286': 'Caramut',
    '3287': 'Glenthompson',
    '3314': 'Glenthompson',
    '3319': 'Serviceton',
    '3329': 'Teesdale',
    '3332': 'Anakie',
    '3333': 'Steiglitz',
    '3334': 'Morrisons',
    '3392': 'Woomelang',
    '3396': 'Underbool',
    '3414': 'Hopetoun',
    '3419': 'Murrayville',
    '3435': 'Trentham',
    '3438': 'Malmsbury',
    '3448': 'Maldon',
    '3451': 'Chewton',
    '3463': 'Dunolly',
    '3468': 'Maryborough',
    '3469': 'Avoca South',
    '3485': 'Quambatook',
    '3498': 'Nyah',
    '3509': 'Ouyen',
    '3517': 'Bendigo North',
    '3520': 'Pyramid Hill',
    '3522': 'Boort',
    '3523': 'Boort',
    '3527': 'Sea Lake',
    '3531': 'Gunbower',
    '3533': 'Pyramid Hill',
    '3546': 'Stanhope',
    '3552': 'Bendigo',
    '3557': 'Long Gully',
    '3561': 'Leitchville',
    '3562': 'Gunbower',
    '3566': 'Echuca West',
    '3567': 'Echuca West',
    '3571': 'Chewton',
    '3573': 'Harcourt North',
    '3590': 'Echuca',
    '3597': 'Piangil',
    '3622': 'Swanpool',
    '3624': 'Bonnie Doon',
    '3662': 'Pyalong',
    '3665': 'Tallarook',
    '3673': 'Swanpool',
    '3682': 'Milawa',
    '3697': 'Falls Creek',
    '3701': 'Corryong North',
    '3705': 'Buchan',
    '3709': 'Mallacoota',
    '3726': 'Cheshunt',
    '3728': 'Oxley',
    '3733': 'Harrietville',
    '3738': 'Mount Beauty',
    '3739': 'Falls Creek',
    '3759': 'Wandong',
    '3763': 'Pyalong',
    '3778': 'St Andrews',
    '3787': 'Kallista',
    '3789': 'Menzies Creek',
    '3796': 'Wesburn',
    '3799': 'Powelltown',
    '3808': 'Pakenham Upper',
    '3809': 'Guys Hill',
    '3814': 'Tynong',
    '3823': 'Yarragon',
    '3824': 'Trafalgar East',
    '3832': 'Traralgon South',
    '3833': 'Tyers',
    '3851': 'Sale East',
    '3862': 'Dargo',
    '3864': 'Buchan',
    '3865': 'Bruthen',
    '3874': 'Welshpool',
    '3886': 'Buchan South',
    '3890': 'Genoa',
    '3909': 'Loch Sport',
    '3918': 'Somerville',
    '3921': 'Morradoo',
    '3923': 'Rhyll',
    '3928': 'Main Ridge',
    '3929': 'Merricks',
    '3931': 'Langwarrin South',
    '3937': 'Baxter',
    '3939': 'Merricks North',
    '3941': 'Morradoo',
    '3943': 'Red Hill South',
    '3944': 'Shoreham',
    '3958': 'Cape Paterson',
    '3959': 'Outtrim',
    '3962': 'Fish Creek',
})




# Outlier price filter (cpl) -- remove clearly bad data entries
PRICE_MIN = {'ulp91': 120, 'ulp95': 130, 'diesel': 150}
PRICE_MAX = {'ulp91': 350, 'ulp95': 370, 'diesel': 420}


#  ARGUMENT PARSER
def parse_args():
    p = argparse.ArgumentParser(description='VIC Suburb-Level Fuel Price Forecast')
    p.add_argument('--api-key', default=os.environ.get('SERVO_SAVER_API_KEY', ''),
                   help='Service Victoria Fair Fuel API Consumer ID')
    p.add_argument('--skip-live', action='store_true',
                   help='Use calibration CSV only, no live API call')
    p.add_argument('--top-n', type=int, default=0,
                   help='Only output top N postcodes by vehicle count (0=all)')
    return p.parse_args()


#  ZONE CLASSIFIER
def classify_zone(postcode):
    try:
        p = int(str(postcode).strip())
        if 3000 <= p <= 3207:   return 'inner_metro'
        elif 3207 < p <= 3500:  return 'middle_metro'
        elif 3500 < p <= 3999:  return 'regional_vic'
        else:                    return 'rural_remote'
    except Exception:
        return 'unknown'


#  LOAD FLEET POSTCODES
def load_fleet_postcodes():
    """
    Load VicRoads fleet data -> per-postcode vehicle counts by fuel type.

    Returns DataFrame columns:
      postcode, suburb_name, zone,
      petrol_vehicles  (P+M: petrol + hybrid  -> ULP91/ULP95 demand proxy)
      diesel_vehicles  (D                     -> Diesel demand proxy)
      hybrid_vehicles  (M: hybrid petrol only)
      ev_vehicles      (E: pure electric)
      total_vehicles   (all registered)
      ev_penetration_pct, diesel_fraction_pct, fleet_type
    """
    csvs = sorted(DAT_DIR.glob('whole_fleet*postcode*.csv'))
    if not csvs:
        raise FileNotFoundError('Cannot find whole_fleet*postcode*.csv in script folder')

    raw = pd.read_csv(csvs[-1])
    raw['CD_CL_FUEL_ENG'] = raw['CD_CL_FUEL_ENG'].astype(str).str.strip()
    raw['POSTCODE']        = raw['POSTCODE'].astype(str).str.strip().str.zfill(4)

    # Aggregate each fuel type per postcode
    petrol = raw[raw['CD_CL_FUEL_ENG'].isin(['P','M'])].groupby('POSTCODE')['TOTAL1'].sum().rename('petrol_vehicles')
    diesel = raw[raw['CD_CL_FUEL_ENG'] == 'D'].groupby('POSTCODE')['TOTAL1'].sum().rename('diesel_vehicles')
    hybrid = raw[raw['CD_CL_FUEL_ENG'] == 'M'].groupby('POSTCODE')['TOTAL1'].sum().rename('hybrid_vehicles')
    ev     = raw[raw['CD_CL_FUEL_ENG'] == 'E'].groupby('POSTCODE')['TOTAL1'].sum().rename('ev_vehicles')
    total  = raw.groupby('POSTCODE')['TOTAL1'].sum().rename('total_vehicles')

    by_pc = pd.concat([petrol, diesel, hybrid, ev, total], axis=1).fillna(0).astype(int)
    by_pc.index.name = 'postcode'
    by_pc = by_pc.reset_index()

    # Keep only postcodes with meaningful fuel vehicle presence
    by_pc = by_pc[(by_pc['petrol_vehicles'] >= 50) | (by_pc['diesel_vehicles'] >= 50)].copy()

    # Derived metrics for each postcode
    by_pc['ev_penetration_pct']  = (
        by_pc['ev_vehicles'] / by_pc['total_vehicles'] * 100).round(1)
    by_pc['diesel_fraction_pct'] = (
        by_pc['diesel_vehicles'] /
        (by_pc['petrol_vehicles'] + by_pc['diesel_vehicles']).clip(lower=1) * 100).round(1)
    by_pc['fleet_type'] = np.where(
        by_pc['diesel_fraction_pct'] >= 40, 'diesel_heavy',
        np.where(by_pc['diesel_fraction_pct'] >= 25, 'mixed', 'petrol_dominant'))

    by_pc['postcode']    = by_pc['postcode'].astype(str).str.zfill(4)
    by_pc['zone']        = by_pc['postcode'].apply(classify_zone)
    by_pc['suburb_name'] = by_pc['postcode'].map(POSTCODE_SUBURBS).fillna('Unknown')

    by_pc = by_pc.sort_values('petrol_vehicles', ascending=False).reset_index(drop=True)

    print(f'  Fleet: {len(by_pc)} postcodes')
    print(f'    Petrol/hybrid vehicles : {by_pc["petrol_vehicles"].sum():>10,}  (ULP91/ULP95 demand base)')
    print(f'    Diesel vehicles        : {by_pc["diesel_vehicles"].sum():>10,}  (Diesel demand base)')
    print(f'    Hybrid vehicles (incl.): {by_pc["hybrid_vehicles"].sum():>10,}  (partial petrol)')
    print(f'    EV vehicles            : {by_pc["ev_vehicles"].sum():>10,}  (no fuel demand)')
    print(f'    Total registered       : {by_pc["total_vehicles"].sum():>10,}')
    return by_pc

def fetch_live_prices(api_key):
    """
    Fetch all VIC station prices from the Fair Fuel Open Data API.
    Returns DataFrame with columns:
      postcode, station_id, station_name, lat, lon, ulp91, ulp95, diesel
    """
    if not api_key:
        print('  No API key -- skipping live fetch')
        return None

    headers = {
        'x-consumer-id':   api_key,
        'x-transactionid': str(uuid.uuid4()),
        'User-Agent':      'VIC-Fuel-Forecast/1.0',
        'Accept':          'application/json',
    }

    try:
        url = API_BASE + '/fuel/prices'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        rows = []
        for station in data.get('fuelPriceDetails', []):
            s = station.get('fuelStation', {})

            # Extract postcode from address string
            # Format: "123 Main St, Melbourne VIC 3000"
            address = s.get('address', '')
            pc_match = re.search(r'\b(3\d{3})\b', address)
            postcode  = pc_match.group(1) if pc_match else None

            if not postcode:
                continue

            row = {
                'postcode':     postcode.zfill(4),
                'station_id':   s.get('id', ''),
                'station_name': s.get('name', ''),
                'brand_id':     s.get('brandId', ''),
                'address':      address,
                'lat':          s.get('location', {}).get('latitude'),
                'lon':          s.get('location', {}).get('longitude'),
            }

            # Extract prices per fuel type
            for fp in station.get('fuelPrices', []):
                ft    = fp.get('fuelType', '')
                price = fp.get('price')
                avail = fp.get('isAvailable', True)
                our   = FUEL_CODES.get(ft)
                if our and price and avail:
                    try:
                        p = float(price)
                        # Apply outlier filter
                        if PRICE_MIN[our] <= p <= PRICE_MAX[our]:
                            row[our] = p
                    except (TypeError, ValueError):
                        pass

            rows.append(row)

        df = pd.DataFrame(rows)
        for fuel in ['ulp91', 'ulp95', 'diesel']:
            if fuel not in df.columns:
                df[fuel] = np.nan

        n_stations = len(df)
        n_with_u91 = df['ulp91'].notna().sum()
        print(f'  Live API: {n_stations} stations fetched, '
              f'{n_with_u91} with ULP91 prices')
        print(f'  Postcodes covered: {df["postcode"].nunique()}')
        return df

    except urllib.error.HTTPError as e:
        print(f'  [API] HTTP {e.code}: {e.reason}')
    except urllib.error.URLError as e:
        print(f'  [API] Connection failed: {e.reason}')
    except Exception as e:
        print(f'  [API] Error: {type(e).__name__}: {e}')

    return None


#  COMPUTE POSTCODE MEDIANS FROM LIVE DATA

def compute_postcode_medians(station_df):
    """
    Group station_df by postcode, compute median price per fuel type.
    Also computes: n_stations, brand_mix (major/independent).
    """
    result_rows = []
    for pc, grp in station_df.groupby('postcode'):
        row = {'postcode': pc, 'n_stations': len(grp)}
        for fuel in ['ulp91', 'ulp95', 'diesel']:
            vals = grp[fuel].dropna()
            if len(vals) > 0:
                row[f'{fuel}_median']  = round(float(np.median(vals)), 1)
                row[f'{fuel}_min']     = round(float(vals.min()), 1)
                row[f'{fuel}_max']     = round(float(vals.max()), 1)
                row[f'{fuel}_n']       = int(len(vals))
            else:
                row[f'{fuel}_median']  = np.nan
                row[f'{fuel}_min']     = np.nan
                row[f'{fuel}_max']     = np.nan
                row[f'{fuel}_n']       = 0
        result_rows.append(row)

    return pd.DataFrame(result_rows)


#  UPDATE CALIBRATION FILE

def update_calibration(postcode_medians, state_medians, run_date):
    """
    Append today's postcode spreads to the calibration CSV.
    spread = postcode_median - state_median
    This builds up a historical record of postcode-level spreads.
    """
    rows = []
    for _, r in postcode_medians.iterrows():
        for fuel in ['ulp91', 'ulp95', 'diesel']:
            pc_med = r.get(f'{fuel}_median')
            st_med = state_medians.get(fuel)
            if pd.notna(pc_med) and st_med and st_med > 0:
                rows.append({
                    'date':      run_date,
                    'postcode':  r['postcode'],
                    'fuel':      fuel,
                    'pc_median': pc_med,
                    'st_median': st_med,
                    'spread':    round(pc_med - st_med, 2),
                    'n_stations':r.get('n_stations', 0),
                })

    new_df = pd.DataFrame(rows)

    if CALIB_FILE.exists():
        old_df = pd.read_csv(CALIB_FILE, dtype={'postcode': str})
        # Keep last 52 weeks per postcode/fuel to avoid unbounded growth
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = (combined.sort_values('date')
                             .groupby(['postcode','fuel'])
                             .tail(52)
                             .reset_index(drop=True))
        combined.to_csv(CALIB_FILE, index=False)
        print(f'  Calibration updated: {len(combined)} records '
              f'({combined["postcode"].nunique()} postcodes)')
    else:
        new_df.to_csv(CALIB_FILE, index=False)
        print(f'  Calibration created: {len(new_df)} records '
              f'({new_df["postcode"].nunique()} postcodes)')

    return new_df


#  LOAD STATE FORECASTS FROM EXISTING MODEL OUTPUTS

def load_state_forecasts():
    """
    Load the state-level forecasts from fuel_price_forecast.py JSON output.
    Falls back to loading model H5 files and re-running if JSON not found.
    """
    json_path = RPT_DIR / 'price_forecast.json'
    if json_path.exists():
        with open(json_path) as f:
            payload = json.load(f)
        forecasts = payload.get('forecasts', {})
        state_fc = {}
        for fuel in ['ulp91', 'ulp95', 'diesel']:
            fc = forecasts.get(fuel, {})
            state_fc[fuel] = {
                'current_cpl':  fc.get('last_actual_cpl'),
                'forecast_cpl': fc.get('pred_price_cpl'),
                'change_cpl':   fc.get('change_cpl'),
                'forecast_date':fc.get('forecast_date'),
                'week_end':     fc.get('week_end'),
            }
        print(f'  State forecasts loaded from price_forecast.json')
        return state_fc
    else:
        print('  price_forecast.json not found -- run fuel_price_forecast.py first')
        return None


#  BUILD SUBURB FORECASTS
def build_suburb_forecasts(postcodes_df, postcode_medians, state_forecasts,
                           state_medians, run_date):
    """
    For every postcode in the fleet dataset, compute a forecast price.

    Logic per postcode per fuel:
      spread = postcode_median - state_median  (from live data or calibration)
      forecast = state_forecast + spread

    Spread source priority:
      1. Live API median (this week)
      2. Calibration CSV historical median spread (past weeks)
      3. Zone-based estimate
    """
    # Load historical calibration
    hist_spreads = {}
    if CALIB_FILE.exists():
        calib = pd.read_csv(CALIB_FILE, dtype={'postcode': str})
        # Use median spread per postcode/fuel over history
        for (pc, fuel), grp in calib.groupby(['postcode', 'fuel']):
            hist_spreads[(pc, fuel)] = float(grp['spread'].median())

    # Build live spread lookup from current API data
    live_spreads = {}
    if postcode_medians is not None and state_medians:
        for _, r in postcode_medians.iterrows():
            pc = str(r['postcode']).zfill(4)
            for fuel in ['ulp91', 'ulp95', 'diesel']:
                pc_med = r.get(f'{fuel}_median')
                st_med = state_medians.get(fuel)
                if pd.notna(pc_med) and st_med:
                    live_spreads[(pc, fuel)] = round(pc_med - st_med, 2)

    forecast_rows = []
    spread_sources = {'live': 0, 'historical': 0, 'zone_model': 0}

    for _, fleet_row in postcodes_df.iterrows():
        pc   = str(fleet_row['postcode']).zfill(4)
        zone = fleet_row['zone']

        suburb_name = str(fleet_row.get('suburb_name', POSTCODE_SUBURBS.get(pc, 'Unknown')))
        petrol_veh = int(fleet_row.get('petrol_vehicles', 0))
        diesel_veh = int(fleet_row.get('diesel_vehicles', 0))
        ev_pct     = float(fleet_row.get('ev_penetration_pct', 0))
        diesel_pct = float(fleet_row.get('diesel_fraction_pct', 0))
        fleet_type = str(fleet_row.get('fleet_type', 'unknown'))
        total_veh  = int(fleet_row.get('total_vehicles', petrol_veh + diesel_veh))

        row = {
            'postcode':            pc,
            'suburb_name':         suburb_name,
            'zone':                zone,
            'petrol_vehicles':     petrol_veh,
            'diesel_vehicles':     diesel_veh,
            'total_vehicles':      total_veh,
            'ev_penetration_pct':  ev_pct,
            'diesel_fraction_pct': diesel_pct,
            'fleet_type':          fleet_type,
            'forecast_date':  state_forecasts['ulp91']['forecast_date'],
            'week_end':       state_forecasts['ulp91']['week_end'],
            'run_date':       run_date,
        }

        for fuel in ['ulp91', 'ulp95', 'diesel']:
            st_fc  = state_forecasts[fuel]['forecast_cpl']
            st_cur = state_forecasts[fuel]['current_cpl']

            if st_fc is None:
                row[f'{fuel}_forecast_cpl'] = None
                row[f'{fuel}_current_cpl']  = None
                row[f'{fuel}_spread_cpl']   = None
                row[f'{fuel}_spread_source']= 'no_state_forecast'
                continue

            # Determine spread
            if (pc, fuel) in live_spreads:
                spread = live_spreads[(pc, fuel)]
                source = 'live'
                spread_sources['live'] += 1
            elif (pc, fuel) in hist_spreads:
                spread = hist_spreads[(pc, fuel)]
                source = 'historical'
                spread_sources['historical'] += 1
            else:
                # Zone model
                zp     = ZONE_SPREADS.get(zone, ZONE_SPREADS['regional_vic'])
                # Add distance-based adjustment for very remote postcodes
                try:
                    p = int(pc)
                    if p >= 3900:   # far east/northeast VIC
                        extra = 5.0
                    elif p >= 3700: # northeast VIC
                        extra = 3.0
                    elif p >= 3600: # central/north VIC
                        extra = 1.0
                    else:
                        extra = 0.0
                except Exception:
                    extra = 0.0
                spread = zp['base'] + extra
                source = 'zone_model'
                spread_sources['zone_model'] += 1

            row[f'{fuel}_current_cpl']   = round(st_cur + spread, 1)
            row[f'{fuel}_forecast_cpl']  = round(st_fc  + spread, 1)
            row[f'{fuel}_change_cpl']    = state_forecasts[fuel]['change_cpl']
            row[f'{fuel}_spread_cpl']    = round(spread, 2)
            row[f'{fuel}_spread_source'] = source

        forecast_rows.append(row)

    df = pd.DataFrame(forecast_rows)
    total = len(df)
    print(f'\n  Spread sources ({total} postcodes):')
    for src, count in spread_sources.items():
        pct = count / (total * 3) * 100   # 3 fuels per postcode
        print(f'    {src:<15}: {count:>6} ({pct:.0f}% of fuel-postcode pairs)')

    return df


#  WRITE SUBURB REPORTS
def write_suburb_reports(suburb_df, state_forecasts, live_source):
    ts = datetime.now().strftime('%d %B %Y %H:%M')
    W  = 72

    #  CSV 
    csv_path = RPT_DIR / 'suburb_price_forecast.csv'
    suburb_df.to_csv(csv_path, index=False)
    print(f'[REPORT] suburb_price_forecast.csv   -> {RPT_DIR}')

    #  JSON (web app payload) 
    class Safe(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.float32, np.float64)): return float(o)
            if isinstance(o, (np.integer,)):                          return int(o)
            if isinstance(o, np.ndarray):                             return o.tolist()
            return super().default(o)

    # Build compact JSON -- one record per postcode
    records = []
    for _, r in suburb_df.iterrows():
        rec = {
            'postcode':            r['postcode'],
            'suburb_name':         r.get('suburb_name', ''),
            'zone':                r['zone'],
            'petrol_vehicles':     int(r.get('petrol_vehicles', 0)),
            'diesel_vehicles':     int(r.get('diesel_vehicles', 0)),
            'total_vehicles':      int(r.get('total_vehicles', 0)),
            'ev_penetration_pct':  r.get('ev_penetration_pct', 0),
            'diesel_fraction_pct': r.get('diesel_fraction_pct', 0),
            'fleet_type':          r.get('fleet_type', ''),
            'forecast_date':       r['forecast_date'],
            'week_end':            r['week_end'],
            'fuels': {}
        }
        for fuel in ['ulp91', 'ulp95', 'diesel']:
            fc_val = r.get(f'{fuel}_forecast_cpl')
            if pd.notna(fc_val):
                rec['fuels'][fuel] = {
                    'current_cpl':   r.get(f'{fuel}_current_cpl'),
                    'forecast_cpl':  fc_val,
                    'change_cpl':    r.get(f'{fuel}_change_cpl'),
                    'spread_cpl':    r.get(f'{fuel}_spread_cpl'),
                    'spread_source': r.get(f'{fuel}_spread_source'),
                }
        records.append(rec)

    payload = {
        'generated_at':  datetime.now().isoformat(),
        'data_source':   live_source,
        'n_postcodes':   len(records),
        'state_forecasts': state_forecasts,
        'suburbs':       records,
    }
    json_path = RPT_DIR / 'suburb_price_forecast.json'
    json_path.write_text(json.dumps(payload, indent=2, cls=Safe), encoding='utf-8')
    print(f'[REPORT] suburb_price_forecast.json  -> {RPT_DIR}')

    #  Text report 
    lines = []
    lines.append('=' * W)
    lines.append('  VIC FUEL PRICE FORECAST  --  SUBURB LEVEL (per postcode)')
    lines.append('=' * W)
    lines.append(f'  Generated  : {ts}')
    lines.append(f'  Data source: {live_source}')
    fc_date = state_forecasts['ulp91']['forecast_date']
    wk_end  = state_forecasts['ulp91']['week_end']
    lines.append(f'  Forecast   : {fc_date} to {wk_end}')
    lines.append(f'  Postcodes  : {len(suburb_df)}')
    lines.append('')

    # State summary
    lines.append('-' * W)
    lines.append('  STATE-LEVEL FORECAST (VIC median, all stations)')
    lines.append('-' * W)
    lines.append(f'  {"Fuel":<10} {"Current cpl":>12} {"Forecast cpl":>13} {"Change":>8}')
    lines.append('  ' + '-' * 46)
    for fuel in ['ulp91', 'ulp95', 'diesel']:
        fc = state_forecasts[fuel]
        chg = fc['change_cpl'] or 0
        lines.append(f'  {fuel.upper():<10} {fc["current_cpl"]:>12.1f} '
                     f'{fc["forecast_cpl"]:>13.1f} {chg:>+8.1f}')

    lines.append('')
    lines.append('-' * W)
    lines.append('  SUBURB FORECASTS  --  ULP91 (Regular Unleaded 91)')
    lines.append('-' * W)
    lines.append(f'  {"Postcode":>9} {"Suburb":<22} {"Zone":<14} {"Petrol Veh":>11} {"Diesel Veh":>11} '
                 f'{"Curr cpl":>9} {"Fcst cpl":>9} {"Spread":>8} {"Src":<10}')
    lines.append('  ' + '-' * (W+12))

    # Sort by petrol_vehicles descending, show top 50
    top = suburb_df.nlargest(50, 'petrol_vehicles')
    for _, r in top.iterrows():
        fc    = r.get('ulp91_forecast_cpl', 'N/A')
        cur   = r.get('ulp91_current_cpl',  'N/A')
        sp    = r.get('ulp91_spread_cpl',   0)
        src   = r.get('ulp91_spread_source','?')[:8]
        name  = str(r.get('suburb_name', POSTCODE_SUBURBS.get(r['postcode'], 'Unknown')))[:20]
        pveh  = int(r.get('petrol_vehicles', 0))
        dveh  = int(r.get('diesel_vehicles', 0))
        if pd.notna(fc):
            lines.append(f'  {r["postcode"]:>9} {name:<22} {r["zone"]:<14} {pveh:>11,} {dveh:>11,} '
                         f'{cur:>9.1f} {fc:>9.1f} {sp:>+8.2f} {src:<10}')

    lines.append(f'  ... ({len(suburb_df)-50} more postcodes in CSV)')

    lines.append('')
    lines.append('-' * W)
    lines.append('  ZONE SUMMARY  --  ULP91 average forecast by zone')
    lines.append('-' * W)
    for zone in ['inner_metro', 'middle_metro', 'regional_vic', 'rural_remote']:
        zdf = suburb_df[suburb_df['zone'] == zone]
        if len(zdf) == 0:
            continue
        vals = zdf['ulp91_forecast_cpl'].dropna()
        if len(vals) == 0:
            continue
        lines.append(f'  {zone:<16}: n={len(zdf):>4}  '
                     f'mean={vals.mean():.1f} cpl  '
                     f'min={vals.min():.1f}  max={vals.max():.1f}')

    lines.append('')
    lines.append('  SPREAD MODEL')
    lines.append('  Suburb forecast = State forecast + Postcode spread')
    lines.append('  Spread source: live (API this week) > historical > zone model')
    lines.append('=' * W)

    report = '\n'.join(lines)
    (RPT_DIR / 'suburb_price_report.txt').write_text(report, encoding='utf-8')
    print(f'[REPORT] suburb_price_report.txt     -> {RPT_DIR}')
    print('\n' + report)

    return report


#  MAIN
def main():
    args = parse_args()
    run_date = datetime.now().strftime('%Y-%m-%d')

    print('=' * 68)
    print('  VIC SUBURB-LEVEL FUEL PRICE FORECAST')
    print(f'  {datetime.now().strftime("%d %B %Y %H:%M")}')
    print('=' * 68)

    #  Step 1: Load fleet postcodes 
    print('\n[STEP 1] Loading fleet postcodes...')
    postcodes_df = load_fleet_postcodes()
    if args.top_n > 0:
        postcodes_df = postcodes_df.head(args.top_n)
        print(f'  Limited to top {args.top_n} postcodes by vehicle count')

    #  Step 2: Fetch live API prices 
    live_source    = 'calibration / zone model (no live data)'
    station_df     = None
    postcode_meds  = None
    state_medians  = {}

    if not args.skip_live and args.api_key:
        print('\n[STEP 2] Fetching live API prices...')
        station_df = fetch_live_prices(args.api_key)

        if station_df is not None and len(station_df) > 0:
            # Compute state-level medians from live data (for spread calculation)
            for fuel in ['ulp91', 'ulp95', 'diesel']:
                vals = station_df[fuel].dropna()
                if len(vals) > 0:
                    state_medians[fuel] = float(np.median(vals))

            print(f'\n  State medians from live data:')
            for fuel, med in state_medians.items():
                print(f'    {fuel.upper()}: {med:.1f} cpl')

            # Compute per-postcode medians
            postcode_meds = compute_postcode_medians(station_df)
            print(f'\n  Postcode medians computed: '
                  f'{len(postcode_meds)} postcodes with station data')

            # Update calibration file
            print('\n[STEP 2b] Updating calibration history...')
            update_calibration(postcode_meds, state_medians, run_date)
            live_source = 'Fair Fuel Open Data API (Service Victoria, 24hr delay)'
    else:
        print('\n[STEP 2] Skipping live API (no key or --skip-live)')

    #  Step 3: Load state-level forecasts 
    print('\n[STEP 3] Loading state-level forecasts...')
    state_forecasts = load_state_forecasts()

    if state_forecasts is None:
        print('  ERROR: Run fuel_price_forecast.py first to generate state forecasts')
        sys.exit(1)

    # Fill state_medians from JSON if not from live API
    if not state_medians:
        for fuel in ['ulp91', 'ulp95', 'diesel']:
            cur = state_forecasts[fuel].get('current_cpl')
            if cur:
                state_medians[fuel] = cur
        print(f'  Using state medians from price_forecast.json:')
        for fuel, med in state_medians.items():
            print(f'    {fuel.upper()}: {med:.1f} cpl')

    #  Step 4: Build suburb forecasts 
    print('\n[STEP 4] Building suburb forecasts...')
    suburb_df = build_suburb_forecasts(
        postcodes_df, postcode_meds, state_forecasts,
        state_medians, run_date)

    print(f'\n  Suburb forecasts complete: {len(suburb_df)} postcodes')

    #  Step 5: Write reports 
    print('\n[STEP 5] Writing reports...')
    write_suburb_reports(suburb_df, state_forecasts, live_source)

    print('\n  ALL DONE')
    print(f'  suburb_price_forecast.csv  -- {len(suburb_df)} postcodes')
    print(f'  suburb_price_forecast.json -- ready for Firebase/Mapbox')

if __name__ == '__main__':
    main()
