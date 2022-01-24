/*
SQLyog Community Edition- MySQL GUI v7.01 
MySQL - 5.0.27-community-nt : Database - resumeverification
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`resumeverification` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `resumeverification`;

/*Table structure for table `user_company_information` */

DROP TABLE IF EXISTS `user_company_information`;

CREATE TABLE `user_company_information` (
  `id` int(11) NOT NULL auto_increment,
  `username` varchar(100) default NULL,
  `company_name` longtext,
  `start_date` varchar(100) default NULL,
  `End_date` varchar(100) default NULL,
  `technology_worked` longtext,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `user_company_information` */

insert  into `user_company_information`(`id`,`username`,`company_name`,`start_date`,`End_date`,`technology_worked`) values (0,'a','Resumeverifier','f','f','f'),(2,'ningesh','Resumeverifier','f','f','f'),(3,'a','Resumeverifier','f','f','f'),(4,'a','Resumeverifier','f','f','f'),(5,'b','Resumeverifier','f','f','f'),(6,'a','Resumeverifier','f','f','f'),(7,'ab','Resumeverifier','f','f','f'),(8,'ab','Resumeverifier','f','f','f'),(9,'a','Resumeverifier','f','f','f'),(10,'a','Resumeverifier','f','f','f'),(11,'ab','Resumeverifier','f','f','f'),(12,'a','Resumeverifier','f','f','f'),(13,'ab','Resumeverifier','f','f','f'),(14,'a','Resumeverifier','f','f','f'),(15,'ab','Resumeverifier','f','f','f'),(16,'a','Resumeverifier','f','f','f'),(17,'a','Resumeverifier','f','f','f'),(18,'a','Resumeverifier','f','f','f'),(19,'ab','Resumeverifier','f','f','f'),(20,'a','Resumeverifier','f','f','f'),(21,'ab','Resumeverifier','f','f','f'),(22,'a','Resumeverifier','f','f','f'),(23,'a','Resumeverifier','f','f','f'),(24,'ab','Resumeverifier','f','f','f'),(25,'ningesh','Resumeverifier','f','f','f'),(26,'admin','Resumeverifier','f','f','f'),(27,'admin','k','k','k','k'),(28,'','','','','');

/*Table structure for table `userdetails` */

DROP TABLE IF EXISTS `userdetails`;

CREATE TABLE `userdetails` (
  `id` int(11) NOT NULL auto_increment,
  `name` varchar(100) default NULL,
  `address` varchar(500) default NULL,
  `mobileno` varchar(20) default NULL,
  `emailid` varchar(100) default NULL,
  `password` varchar(100) default NULL,
  UNIQUE KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `userdetails` */

insert  into `userdetails`(`id`,`name`,`address`,`mobileno`,`emailid`,`password`) values (1,'Sujay','navi mumbai','7894561230','sujay@gmail.com','123'),(2,'Abc','navi mumbai','7894561230','abc@gmail.com','123'),(3,'Sujay','navi mumbai','7894561230','sujay@gmail','123');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
