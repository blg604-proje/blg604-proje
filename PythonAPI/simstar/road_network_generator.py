try:
    import overpy
except ImportError:
    print("pip install overpy")

try:
    import utm
except ImportError:
    print("pip install utm")

from .types import *


class RoadNetworkGenerator():
    def __init__(self):
        self.api = overpy.Overpass()

    def convert_route_to_x_y(self,way):
        offset = utm.from_latlon(float(way.nodes[0].lat), float(way.nodes[0].lon))
        route = []
        for node in way.nodes:
            x = float(node.lat)
            y = float(node.lon)
            utm_value = utm.from_latlon(x, y)
            coordinate = (utm_value[0] - offset[0], utm_value[1] - offset[1])
            route.append(coordinate)
        return route


    def get_way_points(self,track_name=TrackName.DutchGrandPrix):
        way_points = []
        if track_name == TrackName.DutchGrandPrix:
            result = self.api.query("""
                way(around:1500,52.387149, 4.543497)[highway=raceway][wikidata=Q173083];
                (._;>;);
                out body;
                """)
        elif track_name == TrackName.HungaryGrandPrix:
            result = self.api.query("""
                way(around:1500,47.581569, 19.248396)[highway=raceway][wikidata=Q171356];
                (._;>;);
                out body;
                """)
        else:
            return way_points

        if(result.ways):
            for way in result.ways:
                print("Name: %s" % way.tags.get("name", "n/a"))
                print("  Highway: %s" % way.tags.get("highway", "n/a"))
                print("  Nodes:")
                route = self.convert_route_to_x_y(way)
                for i in range(len(route) - 5):
                    way_point = WayPoint(route[i][1], route[i][0])
                    way_points.append(way_point)
        return way_points
