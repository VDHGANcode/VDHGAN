int curl_mvsprintf ( char * buffer , const char * format , va_list ap_save ) {
 int retcode ;
 retcode = dprintf_formatf ( & buffer , storebuffer , format , ap_save ) ;
 * buffer = 0 ;
 return retcode ;
 }