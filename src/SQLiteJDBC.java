
import java.sql.*;

public class SQLiteJDBC {

    public static void main(String args[]) {
        Connection c = null;
        Statement stmt = null;
        try {
            Class.forName("org.sqlite.JDBC");
            c = DriverManager.getConnection("jdbc:sqlite:test.db");

            /*stmt = c.createStatement();
            String sql2 = "DROP TABLE VIassist " ; 
            stmt.executeUpdate(sql2);
            stmt = c.createStatement();
            sql2 = "DROP TABLE Rear " ; 
            stmt.executeUpdate(sql2);
            System.out.println("Opened database successfully"); */

            c.setAutoCommit(true);
            stmt = c.createStatement();
            String sql = "CREATE TABLE VIAssist "
                    + "(ID INT PRIMARY KEY     NOT NULL,"
                    + " A           TEXT    NOT NULL, "
                    + " B            TEXT     NOT NULL, "
                    + " C TEXT NOT NULL)";
            stmt.executeUpdate(sql);
            // stmt = c.createStatement();
            String sql1 = "INSERT INTO VIAssist (ID,A,B,C) "
                    + "VALUES (1,'1','206','Borivli Station' );";
            stmt.executeUpdate(sql1);
            sql1 = "INSERT INTO VIAssist (ID,A,B,C) "
                    + "VALUES (2,'2','296','Saint Francis Institute of Technology' );";
            stmt.executeUpdate(sql1);
            sql1 = "INSERT INTO VIAssist (ID,A,B,C) "
                    + "VALUES (3,'3','204','Mary Immaculate High School' );";
            stmt.executeUpdate(sql1);
            
            sql1 = "INSERT INTO VIAssist (ID,A,B,C) "
                    + "VALUES (4,'4','203','Churchgate' );";
            stmt.executeUpdate(sql1);

            int temp = 2;
            stmt = c.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT * FROM VIAssist;");
            //String  B = rs.getString("B");
            //System.out.println( "Bus  = " + B );
            while (rs.next()) {
                int id = rs.getInt("id");
                String A = rs.getString("A");

                String B = rs.getString("B");
                String C = rs.getString("C");
                // float salary = rs.getFloat("salary");
                System.out.println("ID = " + id);
                System.out.println("Template = " + A);
                System.out.println("Bus  = " + B);
                System.out.println("Destination  = " + C);
                System.out.println();
            }
            rs.close();
            sql = "CREATE TABLE Rear "
                    + "(ID INT PRIMARY KEY     NOT NULL,"
                    + " A           TEXT    NOT NULL, "
                    + " B            TEXT     NOT NULL, "
                    + " C TEXT NOT NULL)";
            stmt.executeUpdate(sql);
            sql1 = "INSERT INTO Rear (ID,A,B,C) "
                    + "VALUES (1,'1','206','Borivli Station' );";
            stmt.executeUpdate(sql1);
            sql1 = "INSERT INTO Rear (ID,A,B,C) "
                    + "VALUES (2,'2','296','Saint Francis Institute of Technology' );";
            stmt.executeUpdate(sql1);
            sql1 = "INSERT INTO Rear (ID,A,B,C) "
                    + "VALUES (3,'3','207','Mary Immaculate High School' );";
            stmt.executeUpdate(sql1);
            sql1 = "INSERT INTO Rear (ID,A,B,C) "
                    + "VALUES (4,'4','203','Churchgate' );";
            stmt = c.createStatement();
             rs = stmt.executeQuery("SELECT * FROM Rear;");
            //String  B = rs.getString("B");
            //System.out.println( "Bus  = " + B );
            while (rs.next()) {
                int id = rs.getInt("id");
                String A = rs.getString("A");

                String B = rs.getString("B");
                String C = rs.getString("C");
                // float salary = rs.getFloat("salary");
                System.out.println("ID = " + id);
                System.out.println("Template = " + A);
                System.out.println("Bus  = " + B);
                System.out.println("Destination  = " + C);
                System.out.println();
            }
            rs.close();
            // String sql2 = "DROP TABLE VIassist " ; 
            //stmt.executeUpdate(sql2);
            stmt.close();
            c.close();
        } catch (Exception e) {
            System.err.println(e.getClass().getName() + ": " + e.getMessage());
            System.exit(0);
        }
        System.out.println("Table created successfully");
    }
}