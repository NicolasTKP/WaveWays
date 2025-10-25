import { Port } from "@/types/route";

// Port data extracted from ports.xlsx
export const PORTS_DATASET: Port[] = [
  { id: "BAKAPIT", name: "Bakapit Port", country: "Malaysia", lat: 4.9554726, lng: 118.5956891, coordinates: "4º 57' 19'' N, 118º 35' 44'' E" },
  { id: "DULANG_MARINE_TERMI", name: "Dulang Marine Terminal Port", country: "Malaysia", lat: 5.8, lng: 104.1666667, coordinates: "5º 47' 60'' N, 104º 10' 0'' E" },
  { id: "JOHORE_BAHRU", name: "Johore Bahru Port", country: "Malaysia", lat: 1.4544618, lng: 103.7532646, coordinates: "1º 27' 16'' N, 103º 45' 11'' E" },
  { id: "KLANG_WESTPORTS", name: "Klang, Westports Port", country: "Malaysia", lat: 2.98, lng: 101.38, coordinates: "2º 58' 48'' N, 101º 22' 47'' E" },
  { id: "KUCHING", name: "Kuching Port", country: "Malaysia", lat: 1.5822588, lng: 110.4290771, coordinates: "1º 34' 56'' N, 110º 25' 44'' E" },
  { id: "LABUAN_SABAH", name: "Labuan, Sabah Port", country: "Malaysia", lat: 5.2830762, lng: 115.2500029, coordinates: "5º 16' 59'' N, 115º 15' 0'' E" },
  { id: "LUMUT", name: "Lumut Port", country: "Malaysia", lat: 4.2666667, lng: 100.65, coordinates: "4º 16' 0'' N, 100º 39' 0'' E" },
  { id: "MUAR", name: "Muar Port", country: "Malaysia", lat: 2.0333333, lng: 102.55, coordinates: "2º 1' 59'' N, 102º 32' 60'' E" },
  { id: "PORT_KELANG", name: "Port Kelang", country: "Malaysia", lat: 3.0034284, lng: 101.3826494, coordinates: "3º 0' 12'' N, 101º 22' 57'' E" },
  { id: "SARIKEI", name: "Sarikei Port", country: "Malaysia", lat: 2.1326472, lng: 111.5166693, coordinates: "2º 7' 57'' N, 111º 31' 0'' E" },
  { id: "SIPITANG_SABAH", name: "Sipitang, Sabah Port", country: "Malaysia", lat: 5.0363202, lng: 115.5249044, coordinates: "5º 2' 10'' N, 115º 31' 29'' E" },
  { id: "TAWAU", name: "Tawau Port", country: "Malaysia", lat: 4.2502906, lng: 117.8833008, coordinates: "4º 15' 1'' N, 117º 52' 59'' E" },
  { id: "BINTANGOR", name: "Bintangor Port", country: "Malaysia", lat: 2.1676959, lng: 111.6329875, coordinates: "2º 10' 3'' N, 111º 37' 58'' E" },
  { id: "DUNGUN_KUALA_DUNGUN", name: "Dungun (Kuala Dungun) Port", country: "Malaysia", lat: 4.7833333, lng: 103.4333333, coordinates: "4º 46' 59'' N, 103º 25' 59'' E" },
  { id: "KEMAMAN", name: "Kemaman Port", country: "Malaysia", lat: 4.3, lng: 103.4666667, coordinates: "4º 17' 60'' N, 103º 28' 0'' E" },
  { id: "KOTA_KINABALU", name: "Kota Kinabalu Port", country: "Malaysia", lat: 5.9666667, lng: 116.0666667, coordinates: "5º 57' 59'' N, 116º 4' 0'' E" },
  { id: "KUDAT", name: "Kudat Port", country: "Malaysia", lat: 6.8833333, lng: 116.85, coordinates: "6º 52' 59'' N, 116º 50' 60'' E" },
  { id: "LAHAD_DATU_SABAH", name: "Lahad Datu, Sabah Port", country: "Malaysia", lat: 5.0333333, lng: 118.3333333, coordinates: "5º 1' 59'' N, 118º 19' 59'' E" },
  { id: "LUNDU", name: "Lundu Port", country: "Malaysia", lat: 1.6666667, lng: 109.8666667, coordinates: "1º 40' 0'' N, 109º 52' 0'' E" },
  { id: "PASIR_GUDANG", name: "Pasir Gudang Port", country: "Malaysia", lat: 1.4487987, lng: 103.8799031, coordinates: "1º 26' 55'' N, 103º 52' 47'' E" },
  { id: "PULAU_TIOMAN", name: "Pulau Tioman Port", country: "Malaysia", lat: 2.7833333, lng: 104.1666667, coordinates: "2º 46' 59'' N, 104º 10' 0'' E" },
  { id: "SEMPORNA_SABAH", name: "Semporna, Sabah Port", country: "Malaysia", lat: 4.4942858, lng: 118.6015631, coordinates: "4º 29' 39'' N, 118º 36' 5'' E" },
  { id: "SUNGAI_UDANG", name: "Sungai Udang Port", country: "Malaysia", lat: 2.2513723, lng: 102.1343663, coordinates: "2º 15' 4'' N, 102º 8' 3'' E" },
  { id: "TELOK_ANSON", name: "Telok Anson Port", country: "Malaysia", lat: 4.0333333, lng: 101, coordinates: "4º 1' 59'' N, 101º 0' 0'' E" },
  { id: "BINTULU_SARAWAK", name: "Bintulu, Sarawak Port", country: "Malaysia", lat: 3.1750887, lng: 113.0300903, coordinates: "3º 10' 30'' N, 113º 1' 48'' E" },
  { id: "GEORGETOWN", name: "Georgetown Port", country: "Malaysia", lat: 5.439908, lng: 100.3046816, coordinates: "5º 26' 23'' N, 100º 18' 16'' E" },
  { id: "KERTEH", name: "Kerteh Port", country: "Malaysia", lat: 4.31, lng: 103.27, coordinates: "4º 18' 35'' N, 103º 16' 11'' E" },
  { id: "KUAH", name: "Kuah Port", country: "Malaysia", lat: 6.3, lng: 99.8568671, coordinates: "6º 18' 0'' N, 99º 51' 24'' E" },
  { id: "KUNAK", name: "Kunak Port", country: "Malaysia", lat: 4.6833333, lng: 118.25, coordinates: "4º 40' 59'' N, 118º 15' 0'' E" },
  { id: "LIMBANG_SARAWAK", name: "Limbang, Sarawak Port", country: "Malaysia", lat: 4.8218954, lng: 115.07097, coordinates: "4º 49' 18'' N, 115º 4' 15'' E" },
  { id: "MIRI", name: "Miri Port", country: "Malaysia", lat: 4.338146, lng: 113.9790277, coordinates: "4º 20' 17'' N, 113º 58' 44'' E" },
  { id: "PENANG", name: "Penang Port", country: "Malaysia", lat: 5.2881403, lng: 100.4184794, coordinates: "5º 17' 17'' N, 100º 25' 6'' E" },
  { id: "SANDAKAN_SABAH", name: "Sandakan, Sabah Port", country: "Malaysia", lat: 5.8546615, lng: 118.1085205, coordinates: "5º 51' 16'' N, 118º 6' 30'' E" },
  { id: "SIBU_SARAWAK", name: "Sibu, Sarawak Port", country: "Malaysia", lat: 2.2855278, lng: 111.829834, coordinates: "2º 17' 7'' N, 111º 49' 47'' E" },
  { id: "TANJUNG_MANIS", name: "Tanjung Manis Port", country: "Malaysia", lat: 2.1551462, lng: 111.3407305, coordinates: "2º 9' 18'' N, 111º 20' 26'' E" },
  { id: "TELOK_RAMUNIA", name: "Telok Ramunia Port", country: "Malaysia", lat: 1.3706794, lng: 104.2467253, coordinates: "1º 22' 14'' N, 104º 14' 48'' E" },
  { id: "BUTTERWORTH", name: "Butterworth Port", country: "Malaysia", lat: 5.25, lng: 100.24, coordinates: "5º 15' 0'' N, 100º 14' 23'' E" },
  { id: "JOHOR", name: "Johor Port", country: "Malaysia", lat: 1.435221, lng: 103.8999975, coordinates: "1º 26' 6'' N, 103º 53' 59'' E" },
  { id: "KLANG_NORTHPORT", name: "Klang, Northport Port", country: "Malaysia", lat: 2.98, lng: 101.38, coordinates: "2º 58' 48'' N, 101º 22' 47'' E" },
  { id: "KUANTAN_TANJONG_GEL", name: "Kuantan (Tanjong Gelang) Port", country: "Malaysia", lat: 3.8, lng: 103.3333333, coordinates: "3º 47' 60'' N, 103º 19' 59'' E" },
  { id: "LABUAN", name: "Labuan Port", country: "Malaysia", lat: 5.2229052, lng: 115.891385, coordinates: "5º 13' 22'' N, 115º 53' 28'' E" },
  { id: "LINGGA_SARAWAK", name: "Lingga, Sarawak Port", country: "Malaysia", lat: 1.35, lng: 111.1666667, coordinates: "1º 21' 0'' N, 111º 10' 0'' E" },
  { id: "MIRI_SARAWAK", name: "Miri, Sarawak Port", country: "Malaysia", lat: 4.4463174, lng: 114.005119, coordinates: "4º 26' 46'' N, 114º 0' 18'' E" },
  { id: "PORT_DICKSON", name: "Port Dickson", country: "Malaysia", lat: 2.5242125, lng: 101.7913884, coordinates: "2º 31' 27'' N, 101º 47' 28'' E" },
  { id: "SAPANGAR_BAY_OIL_TE", name: "Sapangar Bay Oil Terminal Port", country: "Malaysia", lat: 6.1079129, lng: 116.1111668, coordinates: "6º 6' 28'' N, 116º 6' 40'' E" },
  { id: "SIPITANG", name: "Sipitang Port", country: "Malaysia", lat: 5.0983798, lng: 115.5623611, coordinates: "5º 5' 54'' N, 115º 33' 44'' E" },
  { id: "TANJUNG_PELEPAS", name: "Tanjung Pelepas Port", country: "Malaysia", lat: 1.3558965, lng: 103.5483657, coordinates: "1º 21' 21'' N, 103º 32' 54'' E" },
  { id: "TELUK_EWA", name: "Teluk Ewa Port", country: "Malaysia", lat: 6.4224162, lng: 99.7666676, coordinates: "6º 25' 20'' N, 99º 46' 0'' E" }
];

export const searchPorts = (query: string): Port[] => {
  const lowercaseQuery = query.toLowerCase();
  return PORTS_DATASET.filter(
    port =>
      port.name.toLowerCase().includes(lowercaseQuery) ||
      port.country.toLowerCase().includes(lowercaseQuery) ||
      port.id.toLowerCase().includes(lowercaseQuery)
  );
};

export const getPortById = (id: string): Port | undefined => {
  return PORTS_DATASET.find(port => port.id === id);
};

export const getAllPorts = (): Port[] => {
  return PORTS_DATASET;
};
