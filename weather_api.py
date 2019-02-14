import pyowm

owm = pyowm.OWM('f1147045a3b5c94f7bc496789168c3f6')  


forecast = owm.daily_forecast("Milan,it")
tomorrow = pyowm.timeutils.tomorrow()
forecast.will_be_sunny_at(tomorrow)  


observation = owm.weather_at_place('Ghaziabad')
w = observation.get_weather()
print(w)                      
                             


w.get_wind()                  
w.get_humidity()              
print(w.get_temperature('celsius'))  



observation_list = owm.weather_around_coords(-22.57, -43.12)